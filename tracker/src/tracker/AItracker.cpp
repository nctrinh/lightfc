#include "AItracker.h"


static std::vector<cv::Point2f> get_ctr_plane(int score_size, int score_offset, int total_stride) {
    std::vector<cv::Point2f> center_points;
    int fm_width = score_size;
    int fm_height = score_size;
    for (int i=0; i < fm_height; i++) {
        float _y_ctr = float(i) * float(total_stride) + float(score_offset);
        for (int j=0; j < fm_width; j++) {
            center_points.push_back(cv::Point2f(float(j) * float(total_stride) + float(score_offset), _y_ctr) );
        }
    }
    return center_points;
}

AITracker::AITracker(std::vector<std::string> model_paths) {
    this->model_paths = model_paths;
    this->xy_ctr = get_ctr_plane(params.score_size, params.score_offset, params.total_stride);

    input23 = new float*[2];
    input23[0] = new float[XCORR_OUT_HEIGHT*XCORR_IN_WIDTH*FILTER_NUMBER];
    input23[1] = new float[XCORR_OUT_HEIGHT*XCORR_IN_WIDTH*FILTER_NUMBER];
}

InitState AITracker::init() {

    if (this->create_DPU_Cores(model_paths) == false) {
        return InitState::Failure;
    }

    /*Get output of dpu core*/
    this->kernel = this->trackers[0].getOutputData();
    this->output1 = this->trackers[1].getOutputData();
    this->output2 = this->trackers[2].getOutputData();
    this->output3 = this->trackers[3].getOutputData()[0];

    if (this->kernel == NULL || this->output1 == NULL || this->output2 == NULL || this->output3 == NULL) {
        return InitState::Failure;
    }

    return InitState::Success;
}

void AITracker::initialize(cv::Mat img, cv::Rect2f box) {
    cv::Mat img_crop;
    this->initPreprocess(img, box, img_crop);
    this->runInitModel(img_crop);
}

bool AITracker::create_DPU_Cores(std::vector<std::string> model_paths, int timeout) {
    /*Create dpu init tracker runner*/
    /*Create dpu tracker runners*/
    if(model_paths.size() != TRACKING_MODELS){
        return false;
    }

    for (size_t i=0; i < model_paths.size(); i++) {
        this->trackers[i].Init(model_paths.at(i), timeout);
        this->inShapes.push_back(this->trackers[i].getInShapes());
        this->outShapes.push_back(this->trackers[i].getOutShapes());
    }

    return true;
}


void get_subwindow_tracking(cv::Mat img, cv::Mat &img_crop, cv::Point2f center_pos,
                            float model_sz, float original_sz, cv::Scalar& avg_chans)
{
    cv::Rect2f crop = {center_pos.x - original_sz/2, center_pos.y - original_sz/2, original_sz, original_sz};

    float M_13 = crop.tl().x;
    float M_23 = crop.tl().y;
    float M_11 = (crop.br().x - M_13 ) / (model_sz - 1);
    float M_22 = (crop.br().y - M_23) / (model_sz - 1);
    cv::Mat mat2x3 = (cv::Mat_<double>(2, 3) << M_11, 0, M_13, 0, M_22, M_23);

    cv::warpAffine(img, img_crop, mat2x3,
                   cv::Size(int(model_sz), int(model_sz)),
                   cv::INTER_LINEAR + cv::WARP_INVERSE_MAP,
                   cv::BORDER_CONSTANT,
                   avg_chans);
}

void get_subwindow(cv::Mat img, cv::Mat &img_crop, cv::Point2f center,int model_sz,int original_sz,cv::Scalar& channel_average){
    int sz = original_sz;
    int rows_num = img.rows;
    int cols_num = img.cols;
    float cxy =(float) (original_sz + 1) / 2;
    double context_xmin = std::floor(center.x - cxy + 0.5);
    double context_xmax = context_xmin + sz - 1;
    double context_ymin = std::floor(center.y - cxy + 0.5);
    double context_ymax = context_ymin + sz - 1;
    int left_pad = int(std::max(0.0, -context_xmin));
    int top_pad = int(std::max(0., -context_ymin));
    int right_pad = int(std::max(0., context_xmax - cols_num + 1));
    int bottom_pad = int(std::max(0., context_ymax - rows_num + 1));
    context_xmin = context_xmin + left_pad;
    context_xmax = context_xmax + left_pad;
    context_ymin = context_ymin + top_pad;
    context_ymax = context_ymax + top_pad;
    cv::Mat img_path;
    cv::Rect rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1);
    if ((top_pad + left_pad + bottom_pad + right_pad) > 0){
        int new_rows = rows_num + top_pad + bottom_pad;
        int new_cols = cols_num + left_pad + right_pad;
        cv::Mat tmpl_img(new_rows,new_cols,CV_32FC1);
        cv::copyMakeBorder(img,tmpl_img,top_pad,bottom_pad,left_pad,right_pad,cv::BORDER_CONSTANT,channel_average);
        img_path = tmpl_img(rect);
    }else{
        img_path = img(rect);
    }
    cv::resize(img_path,img_crop,cv::Size(model_sz,model_sz));
}

void get_crop(cv::Mat img, cv::Mat &img_crop, float &scale, cv::Point2f center_pos,
              cv::Size2f target_sz, float z_size, cv::Scalar& avg_chans, float context_amount,
              float x_size, float output_size)
{
    float wc = target_sz.width + context_amount * (target_sz.width + target_sz.height);
    float wh = target_sz.height + context_amount * (target_sz.width + target_sz.height);

    float s_crop = std::sqrt(wc * wh);
    scale = z_size / s_crop;

    if (x_size == 0.f) x_size = z_size;
    if (output_size == 0.f) output_size = x_size;

    s_crop = x_size / scale;
    get_subwindow_tracking(img, img_crop, center_pos, output_size, s_crop, avg_chans);
}

void AITracker::initPreprocess(cv::Mat img, cv::Rect2f box, cv::Mat &img_crop) {
    this->center_pos = (box.br() + box.tl())*0.5;
    this->target_sz = box.size();
    this->avg_chans = cv::mean(img);
    int z_size = params.z_size;
    float context_amount = params.context_amount;
    float scale;
    get_crop(img, img_crop, scale, this->center_pos, this->target_sz,
             z_size, this->avg_chans, context_amount);
}

static inline float sigmoid(float x) {
    return 1/(1+ std::exp(-x));
}
static inline float cal_sz(float w,float h){
    float pad = (w + h) * 0.5f;
    float res = (w + pad) * (h + pad);
    return std::sqrt(res);
}

static inline float change(float r){
    return std::max(r,1.0f/r);
}

static inline void restrict_box(cv::Point2f &center_pos, cv::Size2f &target_sz, cv::Size2f boudary, float min_h, float min_w){
    center_pos.x = std::max(0.0f, std::min(center_pos.x, boudary.width));
    center_pos.y = std::max(0.0f, std::min(center_pos.y, boudary.height));
    target_sz.width = std::max(min_h, std::min(target_sz.width, boudary.width));
    target_sz.height = std::max(min_w, std::min(target_sz.height, boudary.height));
}

void AITracker::preprocess(cv::Mat img, cv::Mat &x_crop, float& x_scale) {
    int z_size = params.z_size;
    int x_size = params.x_size;
    float context_amount = params.context_amount;

    get_crop(img, x_crop, x_scale, this->center_pos, this->target_sz,
             z_size, this->avg_chans, context_amount, x_size);
}

void AITracker::runInitModel(cv::Mat &zcrop) {
    cv::Mat float_img(zcrop.rows, zcrop.cols, CV_32FC1);
    zcrop.convertTo(float_img,CV_32FC1,1,0);
    float** DataInputs = new float*[1];
    DataInputs[0] = float_img.ptr<float>(0);
    this->trackers[0].Run(DataInputs);

    delete [] DataInputs;
}

void AITracker::runTrackModel(size_t index, float* input_data) {
    float** DataInputs = new float*[1];
    DataInputs[0] = input_data;
    this->trackers[index].Run(DataInputs);

    delete [] DataInputs;
}

bool AITracker::postprocess(cv::Mat img, float* cls_score, float* ctr_score, float* offsets, size_t size) {
    float* score = new float[size];
    for (size_t i=0; i < size; i++) {
        cls_score[i] = sigmoid(cls_score[i]);
        ctr_score[i] = sigmoid(ctr_score[i]);
        score[i] = cls_score[i] * ctr_score[i];
    }

    for (size_t i=0; i < size*4; i++) {
        offsets[i] = std::exp(params.si * offsets[i] + params.bi) * params.total_stride;
    }

    std::vector<cv::Rect2f> pred_boxs;
    for (size_t i=0; i < xy_ctr.size(); i++) {
        cv::Point2f xy0(xy_ctr.at(i).x - offsets[i*4], xy_ctr.at(i).y - offsets[i*4+1]);
        cv::Point2f xy1(xy_ctr.at(i).x + offsets[i*4+2], xy_ctr.at(i).y + offsets[i*4+3]);
        pred_boxs.push_back(cv::Rect2f(xy0, xy1));
    }

    /*post process score*/
    cv::Size2f target_sz_in_crop(this->target_sz.width * this->scale, this->target_sz.height * this->scale);

    size_t best_pscore_id = 0;
    float max_pscore = 0.0f;
    float best_penalty = 0.0f;
    for (size_t i=0; i < size; i++) {
        // scale penalty
        float s_c = change(cal_sz(pred_boxs.at(i).width, pred_boxs.at(i).height) / cal_sz(target_sz_in_crop.width , target_sz_in_crop.height));
        // ratio penalty
        float r_c = change((target_sz_in_crop.width / target_sz_in_crop.height) / (pred_boxs.at(i).width / pred_boxs.at(i).height));

        float penalty = std::exp(-(r_c * s_c -1) * params.penalty_k);
        float pscore = penalty * score[i] * (1 - params.window_influence) + window.at(i) * params.window_influence;

        if (pscore > max_pscore) {
            best_pscore_id = i;
            best_penalty = penalty;
            max_pscore = pscore;
        }
    }

    /*post process box*/
    cv::Rect2f pred_in_crop = pred_boxs[best_pscore_id];
    float lr = best_penalty * score[best_pscore_id] * params.test_lr;
    cv::Point2f center = (pred_in_crop.br() + pred_in_crop.tl())*0.5 / this->scale;
    cv::Size2f pred_crop_sz = pred_in_crop.size() / this->scale;

    float res_x = center.x + this->center_pos.x - (params.x_size / 2) / this->scale;
    float res_y = center.y + this->center_pos.y - (params.x_size / 2) / this->scale;
    float res_w = this->target_sz.width * (1 - lr) + pred_crop_sz.width * lr;
    float res_h = this->target_sz.height * (1 - lr) + pred_crop_sz.height * lr;

    this->center_pos = cv::Point2f(res_x, res_y);
    this->target_sz = cv::Size2f(res_w, res_h);

    restrict_box(this->center_pos, this->target_sz, cv::Size2f(img.cols, img.rows), params.min_h, params.min_w);

    this->rect = cv::Rect2f(this->center_pos.x - this->target_sz.width / 2,
                      this->center_pos.y - this->target_sz.height / 2,
                      this->target_sz.width, this->target_sz.height);

    delete [] score;
    if (max_pscore > this->params.loss_track_threshold) {
        if (this->loss_track_counter > 0) {
           this->reset_loss_track_counter++;
           if (this->reset_loss_track_counter > params.reset_count_loss_track_threshold) {
               this->loss_track_counter = 0;
               this->reset_loss_track_counter = 0;
           }
        }

    }
    else {
        this->loss_track_counter++;
        this->reset_loss_track_counter = 0;
        if (this->loss_track_counter > params.loss_track_timeout) {
            return false;
        };
    }
    return true;
}

// convolution2d with input and kernel are ordered in channnel last
// output is ordered in channel last
void conv2d_channel_first(float* input, float* kernel, float* output) {
    float* c_first_input = new float[XCORR_IN_WIDTH*XCORR_IN_HEIGHT];
    float* c_first_kernel = new float[KERNEL_WIDTH*KERNEL_HEIGHT];

    // transpose a kernel of list kernels to first channel kernel
    for (int kh = 0; kh < KERNEL_HEIGHT; kh++) {
        for (int kw = 0; kw < KERNEL_WIDTH; kw++) {
            c_first_kernel[KERNEL_WIDTH*kh + kw] = kernel[FILTER_NUMBER*kh*KERNEL_WIDTH + FILTER_NUMBER*kw];
        }
    }

    // transpose an input of list inputs to first channel kernel
    for (int ih = 0; ih < XCORR_IN_HEIGHT; ih++) {
        for (int iw = 0; iw < XCORR_IN_WIDTH; iw++) {
            c_first_input[XCORR_IN_WIDTH*ih + iw] = input[FILTER_NUMBER*ih*XCORR_IN_WIDTH + FILTER_NUMBER*iw];
        }
    }

    // convolution of an input with a kernel
    for (int ih = 0; ih < XCORR_OUT_HEIGHT; ih++) {
        for (int iw = 0; iw < XCORR_OUT_WIDTH; iw++) {
            float sum_pixel = 0.0f;
            int k_index = 0;
            for (int kh = ih; kh < ih + KERNEL_HEIGHT; kh++) {
                for (int kw = iw; kw < iw + KERNEL_WIDTH; kw++) {
                    sum_pixel += c_first_input[kh*XCORR_IN_WIDTH + kw] * c_first_kernel[k_index++];
                }
            }
            output[FILTER_NUMBER*(XCORR_OUT_WIDTH*ih + iw)] = sum_pixel;
        }
    }

    delete[] c_first_input;
    delete[] c_first_kernel;
}

class ParallelXcorr_ProcessCFirst: public cv::ParallelLoopBody {
private:
    float **input;
    float **kernel;
    float **output;
public:
    ParallelXcorr_ProcessCFirst(float** _input, float** _kernel, float** _output):
        input(_input), kernel(_kernel), output(_output)
    {
    }

    virtual void operator ()(const cv::Range& range) const
    {
        for (int c = range.start; c < range.end; c++) {
            int ind = c < FILTER_NUMBER ? 0 : 1;
            int chan = c%FILTER_NUMBER;

            conv2d_channel_first(input[ind] + chan, kernel[ind] + chan, output[ind] + chan);
        }
    }

    ParallelXcorr_ProcessCFirst& operator=(const ParallelXcorr_ProcessCFirst &) {
        return *this;
    }
};

void AITracker::xcorr_depthwise() {
    ParallelXcorr_ProcessCFirst parallelXcorr_process(output1, kernel, input23);
    cv::parallel_for_(cv::Range(0, 2*FILTER_NUMBER), parallelXcorr_process);
}

TrackResult AITracker::update(cv::Mat img) {
    cv::Mat x_crop;
    this->preprocess(img, x_crop, this->scale);

    cv::Mat float_img(x_crop.rows, x_crop.cols, CV_32FC1);
    x_crop.convertTo(float_img, CV_32FC1, 1, 0);

    this->runTrackModel(1, float_img.ptr<float>(0));

    this->xcorr_depthwise();

    this->runTrackModel(2, input23[1]);

    this->runTrackModel(3, input23[0]);

    bool track_state = this->postprocess(img, output2[0], output2[1], output3, outShapes.at(2)[0].size);

    TrackResult result;

    if(track_state == true){
        result.trackState_ = TrackerState::Tracking;
        result.rect_ = this->rect;
    }else{
        result.trackState_ = TrackerState::Lost;
        result.rect_ = this->rect;
    }

    return result;
}

AITracker::~AITracker() {
    delete [] input23[0];
    delete [] input23[1];
    delete [] input23;
}
