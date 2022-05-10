#include "utils.hpp"

#include <iostream>
#include <opencv2/ml.hpp>

cv::Mat3b quantify_image(const cv::Mat3b& source, int q_colors) {
  cv::Mat points;
  source.convertTo(points, CV_32F);
  points = points.reshape(3, source.rows * source.cols);
  cv::TermCriteria condition(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                             20, 1.0);
  cv::Mat labels;
  cv::Mat centers;
  cv::kmeans(points, q_colors, labels, condition, 10, cv::KMEANS_RANDOM_CENTERS,
             centers);

  for (int i = 0; i < points.rows; i++) {
    points.at<cv::Point3f>(i, 0) =
        centers.at<cv::Point3f>(labels.at<int>(i, 0), 0);
  }
  points = points.reshape(3, source.rows);
  points.convertTo(points, CV_8U);

  return points;
}
