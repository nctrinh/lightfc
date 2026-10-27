#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <algorithm>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <time.h>

#include "AItracker.h"

using namespace std;

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const& path, vector<string>& images) {
    images.clear();
    struct dirent* entry;

    /*Check if path is a valid directory path. */
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }

    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
        exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            string name = entry->d_name;
            string ext = name.substr(name.find_last_of(".") + 1);
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
              (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
                images.push_back(name);
            }
        }
    }

    closedir(dir);
}

/**
 * @brief parse "x,y,w,h" (separators: ',', tab or space; float values, e.g. GOT-10k/LaSOT/OTB)
 */
bool Str2Rect(std::string text, cv::Rect2f &rect) {
    for (auto& ch : text) {
        if (ch == ',' || ch == '\t') ch = ' ';
    }
    std::istringstream ss(text);
    float v[4];
    for (int i = 0; i < 4; i++) {
        if (!(ss >> v[i])) return false;
    }
    rect = cv::Rect2f(v[0], v[1], v[2], v[3]);
    return true;
}

void LoadRect(string const& path, std::vector<cv::Rect2f>& bboxs) {
    std::ifstream filebbox(path);
    std::string info;

    while (std::getline(filebbox, info)) {
        cv::Rect2f rect;
        if (!Str2Rect(info, rect)) {
            rect = cv::Rect2f(NAN, NAN, NAN, NAN);  // keep frame alignment for unparsable / NaN lines
        }
        bboxs.push_back(rect);
    }

    filebbox.close();
}


int main(int argc, char* argv[])
{
    // Check args
    if (argc < 4) {
        cout << "Usage: ./TrackerV2 [config_file] [input_folder_path] [file_box_path] [output_folder_path]" << endl;
        cout << "  config_file: key=value file, see config/lightfc.cfg" << endl;
        return -1;
    }

    string configPath = argv[1];
    string baseImagePath = argv[2];
    string groundtruthbox = argv[3];

    string outputPath = "";

    bool save = false;
    if (argc == 5) {
        outputPath = argv[4];
        save = true;
    }

    LightFCConfig cfg;
    string err;
    if (!cfg.load(configPath, err)) {
        cerr << "Config error: " << err << endl;
        return 1;
    }

    AITracker tracker(cfg);
    if (tracker.init() != InitState::Success) {
        cerr << "Tracker init failed: " << tracker.lastError() << endl;
        return 1;
    }
    vector<string> list_images;
    std::vector<cv::Rect2f> bboxs;

    ListImages(baseImagePath, list_images);
    std::sort(list_images.begin(), list_images.end());

    LoadRect(groundtruthbox, bboxs);
    if (list_images.empty() || bboxs.empty() || std::isnan(bboxs[0].width)) {
        cerr << "No images or no valid initial box" << endl;
        return 1;
    }

    // Frames are read in colour (BGR) and converted inside the tracker with cvtColor, like lib_orin
    // (cv.COLOR_RGB2GRAY). IMREAD_GRAYSCALE would use the JPEG decoder's own gray conversion.
    cv::Mat firstImg = cv::imread(baseImagePath + "/" + list_images[0], cv::IMREAD_COLOR);
    tracker.initialize(firstImg, bboxs[0]);

    std::ofstream results;
    if (save) {
        results.open(outputPath + "/results.txt");
        const cv::Rect2f& b = bboxs[0];
        results << b.x << "," << b.y << "," << b.width << "," << b.height << ",1" << "\n";
    }

    float tracker_time = 0;

    for (size_t i=1; i<list_images.size(); i++) {
        cv::Mat img = cv::imread(baseImagePath + "/" + list_images[i], cv::IMREAD_COLOR);
        auto start = GET_MOMENT;
        TrackResult track_result = tracker.update(img);

        cv::Rect2f new_pos = track_result.rect_;
        auto stop = GET_MOMENT;
        auto duration = GET_DURATION(stop, start);

        tracker_time += duration/1000.f;

        if (save) {
            results << new_pos.x << "," << new_pos.y << "," << new_pos.width << "," << new_pos.height << ","
                    << track_result.score_ << "\n";
        }

        if (!save && !cfg.show) continue;

        // Drawing the ground truth bounding box
        if (i < bboxs.size() && !std::isnan(bboxs[i].width)) {
            cv::rectangle(img, bboxs[i].tl(), bboxs[i].br(),
                          cv::Scalar(0, 255, 0),
                          1, cv::LINE_8);
        }

        // Drawing the predict bounding box
        cv::rectangle(img, new_pos.tl(), new_pos.br(),
                      cv::Scalar(0, 255, 255),
                      1, cv::LINE_8);

        if (track_result.trackState_ == TrackerState::Lost) {
            std::cout << "Lost track at image " << list_images[i] << std::endl;
            cv::putText(img, "Lost", cv::Point(30, 30), 0, 1, cv::Scalar(0, 0, 255));
        }

        if (save) {
            cv::imwrite(outputPath + "/" + list_images[i], img);
        }

        if (cfg.show) {
            cv::imshow("Output", img);
            cv::waitKey(1);
        }
    }

    std::cout << "FPS: " << (list_images.size() -1) * 1000 / tracker_time << std::endl;
    std::cout << "Time process: " << tracker_time << std::endl;

    return 0;
}
