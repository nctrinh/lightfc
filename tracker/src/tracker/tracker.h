#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>

enum class InitState : unsigned char{
    Success,
    Failure,
};
enum class TrackerState : unsigned char{
    Tracking,
    Lost
};

struct TrackResult
{
    TrackerState trackState_;
    cv::Rect2f rect_;
    float score_ = 0.f;     // model confidence (LightFC: max of the ctr score map)
};

class Tracker
{
public:
    Tracker();
    virtual ~Tracker();
    virtual InitState init() = 0;
    virtual void initialize(cv::Mat img, cv::Rect2f roi) = 0;
    virtual TrackResult update(cv::Mat img) = 0;
};

#endif // TRACKER_H
