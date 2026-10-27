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

void Str2Array(std::string text, std::vector<int> &numbers) {
    size_t pos = 0;
    while ((pos = text.find(',')) != std::string::npos) {
        numbers.push_back(std::stoi(text.substr(0, pos)));
        text.erase(0, pos + 1);
    }
    numbers.push_back(std::stoi(text.substr(0, pos)));
}

void LoadRect(string const& path, std::vector<cv::Rect2f>& bboxs) {
    std::ifstream filebbox(path);
    std::string info;

    while (std::getline(filebbox, info)) {
        std::vector<int> bbox;
        Str2Array(info, bbox);
        bboxs.push_back(cv::Rect2f(bbox[0], bbox[1], bbox[2], bbox[3]));
    }

    filebbox.close();
}


int main(int argc, char* argv[])
{
    // Check args
    if (argc < 4) {
        cout << "Usage of deblur demo: ./Tracker [model_path] [input_folder_path] [file_box_path] [output_folder_path]" << endl;
        return -1;
    }

    string modelPath = argv[1];
    string baseImagePath = argv[2];
    string groundtruthbox = argv[3];

    string outputPath = "";

    bool show = true;
    bool save = false;
    if (argc == 5) {
        outputPath = argv[4];
        save = true;
    }

    AITracker tracker({modelPath + "/Init_model.xmodel", modelPath + "/Track_model1.xmodel",
                     modelPath + "/Track_model2.xmodel", modelPath + "/Track_model3.xmodel"});
    tracker.init();
    vector<string> list_images;
    std::vector<cv::Rect2f> bboxs;

    ListImages(baseImagePath, list_images);
    std::sort(list_images.begin(), list_images.end());

    LoadRect(groundtruthbox, bboxs);

    cv::Mat firstImg = cv::imread(baseImagePath + "/" + list_images[0], cv::IMREAD_GRAYSCALE);
    tracker.initialize(firstImg, bboxs[0]);

    float tracker_time = 0;

    for (size_t i=1; i<list_images.size(); i++) {
        cv::Mat img = cv::imread(baseImagePath + "/" + list_images[i], cv::IMREAD_GRAYSCALE);
        auto start = GET_MOMENT;
        TrackResult track_result = tracker.update(img);

        cv::Rect2f new_pos = track_result.rect_;
        auto stop = GET_MOMENT;
        auto duration = GET_DURATION(stop, start);

        tracker_time += duration/1000;

        cv::cvtColor(img, img,  cv::COLOR_GRAY2RGB);
        // Drawing the ground truth bounding box
        cv::rectangle(img, bboxs[i].tl(), bboxs[i].br(),
                      cv::Scalar(0, 255, 0),
                      1, cv::LINE_8);

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

        if (show) {
            cv::imshow("Output", img);
            cv::waitKey(1);
        }
    }

    std::cout << "FPS: " << (list_images.size() -1) * 1000 / tracker_time << std::endl;
    std::cout << "Time process: " << tracker_time << std::endl;

    return 0;
}
