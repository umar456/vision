
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include <iostream>
#include <string>
#include <vector>

struct hmd_frame {
  cv::Mat image;
  cv::Mat pose;
};

std::pair<cv::Matx33f, cv::Vec3f> get_rotation_translation(const cv::Mat &p1) {
  cv::Mat R(3, 3, CV_32F);
  cv::Mat t(3, 1, CV_32F);
  p1(cv::Rect(0, 0, 3, 3)).copyTo(R);
  p1(cv::Rect(3, 0, 1, 3)).copyTo(t);
  return std::make_pair(R, t);
}

cv::Mat get_relative_pose(const cv::Mat &p1, const cv::Mat &p2) {
  return p1.inv() * p2;
}

cv::Matx33f cross_mat(const cv::Vec<float, 3> &vec) {
  cv::Matx33f out;
  out << 0, -vec(2), vec(1), vec(2), 0, -vec(0), -vec(1), vec(0), 0;
  return out;
}

cv::Matx33f essential_matrix(const cv::Mat &p1, const cv::Mat &p2) {
  auto rpose = get_relative_pose(p1, p2);
  auto[R, t] = get_rotation_translation(rpose);
  cv::Matx33f S = cross_mat(t);
  cv::Matx33f e = cv::Matx33f::eye();
  e(0, 0) *= -1;
  e(2, 2) *= -1;

  auto[R1, t1] = get_rotation_translation(p1);
  auto[R2, t2] = get_rotation_translation(p2);
  return S * R;
}

cv::Matx33f fundamental_matrix(const cv::Mat &p1, const cv::Mat &p2,
                               const cv::Mat &cam) {
  auto E = essential_matrix(p1, p2);
  return cv::Mat(cam.inv().t() * cv::Mat(E) * cam.inv());
}

// take number image type number (from cv::Mat.type()), get OpenCV's enum
// string.
std::string getType(int imgTypeInt) {
  int numImgTypes =
      35; // 7 base types, with five channel options each (none or C1, ..., C4)

  int enum_ints[] = {CV_8U,    CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,  CV_8S,
                     CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,  CV_16U,   CV_16UC1,
                     CV_16UC2, CV_16UC3, CV_16UC4, CV_16S,   CV_16SC1, CV_16SC2,
                     CV_16SC3, CV_16SC4, CV_32S,   CV_32SC1, CV_32SC2, CV_32SC3,
                     CV_32SC4, CV_32F,   CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
                     CV_64F,   CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};

  std::string enum_strings[] = {
      "CV_8U",    "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",  "CV_8S",
      "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",  "CV_16U",   "CV_16UC1",
      "CV_16UC2", "CV_16UC3", "CV_16UC4", "CV_16S",   "CV_16SC1", "CV_16SC2",
      "CV_16SC3", "CV_16SC4", "CV_32S",   "CV_32SC1", "CV_32SC2", "CV_32SC3",
      "CV_32SC4", "CV_32F",   "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
      "CV_64F",   "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"};

  for (int i = 0; i < numImgTypes; i++) {
    if (imgTypeInt == enum_ints[i])
      return enum_strings[i];
  }
  return "unknown image type";
}

void bm_demo(cv::viz::Viz3d &win, std::vector<hmd_frame> &hmd_data,
             cv::Mat intrinsics, cv::Size image_size) {
  using namespace cv;
  using namespace std;
  Ptr<StereoSGBM> bm = StereoSGBM::create(0, 16, 5, 8 * 1 * 5 * 5, 32 * 5 * 5,
                                          0, 0, 15, 50, 1, StereoSGBM::MODE_HH);
  // int SADWindowSize = 5;
  int numberOfDisparities = 32;
  int frame_interval = 8;

  while (!win.wasStopped()) {
    vector<Affine3f> path;
    for (int i = 1; i < 200; i++) {
      imshow("img", hmd_data[i].image);
      Mat frame;
      flip(hmd_data[i].image, frame, -1);
      auto cam = viz::WCameraPosition(Matx33d(intrinsics), frame, 0.1);
      auto path_widget = viz::WTrajectory(path);

      Matx33f e = Matx33f::eye();
      e(0, 0) *= -1;
      e(2, 2) *= -1;

      auto[R, t] = get_rotation_translation(hmd_data[i].pose);
      Affine3f pose(R * e, t);

      path.push_back(pose);
      win.showWidget("Camera", cam);
      win.showWidget("Path", path_widget);
      win.setWidgetPose("Camera", pose);

      waitKey(30);
      if (i > frame_interval) {
        auto cpose = hmd_data[i].pose;
        auto ppose = hmd_data[i - frame_interval].pose;
        auto rpose_mat = get_relative_pose(cpose, ppose);
        cout << rpose_mat << endl;

        auto[R, t] = get_rotation_translation(rpose_mat);
        Affine3f rpose(R, t);

        auto pframe = hmd_data[i - frame_interval].image;
        Mat fpframe;
        flip(pframe, fpframe, -1);
        imshow("pframe", pframe);
        auto ccam = viz::WCameraPosition(Matx33d(intrinsics), frame);
        auto pcam = viz::WCameraPosition(Matx33d(intrinsics), fpframe);
        stringstream css;
        css << "Current Camera";
        win.showWidget(css.str(), ccam);
        win.setWidgetPose(css.str(), rpose);
        stringstream pss;
        pss << "Previous Camera";
        win.showWidget(pss.str(), pcam);
        win.setWidgetPose(pss.str(),
                          Affine3f(Mat::eye(3, 3, CV_32F), Vec3f(0, 0, 0)));

        Mat R1(3, 3, CV_32F), R2(3, 3, CV_32F), P1(3, 4, CV_32F),
            P2(3, 4, CV_32F), Q(4, 4, CV_32F);
        intrinsics.convertTo(intrinsics, CV_32F);
        Mat Rmat(3, 3, CV_64F);
        Mat tmat(3, 1, CV_64F);
        Mat(R).convertTo(Rmat, CV_64F);
        Mat(t).convertTo(tmat, CV_64F);

        Rect roi1, roi2;
        stereoRectify(intrinsics, Mat(), intrinsics, Mat(), image_size,
                      Rmat, tmat, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1,
                      image_size, &roi1, &roi2);

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(intrinsics, Mat(), R1, P1, image_size, CV_16SC2,
                                map11, map12);
        initUndistortRectifyMap(intrinsics, Mat(), R2, P2, image_size, CV_16SC2,
                                map21, map22);

        // {
        //   auto[cR, ct] = get_rotation_translation(P1);
        //   Mat rmat;
        //   R1.convertTo(rmat, CV_32F);
        //   auto crectpose = Affine3f(rmat, Vec3f(0, 0, 0));
        //   win.setWidgetPose(css.str(), crectpose);
        // }
        // {
        //   auto[cR, ct] = get_rotation_translation(P2);
        //   Mat rmat;
        //   R2.convertTo(rmat, CV_32F);
        //   auto crectpose = Affine3f(rmat, t);
        //   win.setWidgetPose(pss.str(), crectpose);
        // }

        Mat img1r, img2r;
        flip(frame, frame, -1);
        remap(frame, img1r, map11, map12, INTER_LINEAR);
        remap(pframe, img2r, map21, map22, INTER_LINEAR);
        imshow("img1r", img1r);
        imshow("img2r", img2r);

        // bm->setROI1(roi1);
        // bm->setROI2(roi2);
        bm->setPreFilterCap(31);
        // bm->setPreFilterSize(9);
        // bm->setPreFilterType(StereoBM::PREFILTER_XSOBEL);
        // bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
        // bm->setMinDisparity(0);
        bm->setNumDisparities(numberOfDisparities);
        // bm->setTextureThreshold(10);
        // bm->setUniquenessRatio(15);
        bm->setSpeckleWindowSize(100);
        bm->setSpeckleRange(32);
        // bm->setDisp12MaxDiff(1);
        Mat disp, disp8;
        Mat img1, img2;
        cvtColor(img1r, img1, COLOR_RGB2GRAY);
        cvtColor(img2r, img2, COLOR_RGB2GRAY);
        bm->compute(img1, img2, disp);

        if (numberOfDisparities) {
          disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities * 16.));
        }
        // disp.convertTo(disp8, CV_8U);
        imshow("disparity", disp8);

        Mat xyz;
        reprojectImageTo3D(disp, xyz, Q, false);
        // cout << xyz << endl;

        auto cloud = viz::WCloud(xyz);
        win.showWidget("cloud", cloud);
        win.setWidgetPose("cloud", pose);

        // img1 = img1r;
        // img2 = img2r;
      }
      win.spinOnce(1, true);
    }
  }
}

using cv::FileStorage;
using cv::Mat;
using cv::Matx33f;
using cv::Matx33d;
using cv::Size;
using cv::Rect;
using cv::imread;
using cv::CascadeClassifier;
using cv::Affine3f;
using cv::Ptr;
using cv::StereoSGBM;
using cv::CASCADE_SCALE_IMAGE;
using cv::COLOR_RGB2GRAY;
using cv::viz::WCoordinateSystem;
using cv::viz::WCameraPosition;
using std::vector;
using std::string;
using std::stringstream;
using std::cout;
using std::endl;

int main(int argc, char **argv) {
  FileStorage fs("../../vr_data/pose.yaml", FileStorage::READ);

  Mat intrinsics;
  Size image_size;
  fs["image_size"] >> image_size;
  fs["camera_intrensics"] >> intrinsics;

  cout << image_size << endl;
  cout << intrinsics << endl;

  vector<hmd_frame> hmd_data;

  static const float focal_length = 0.013;
  static const float pixel_size = 1.0*1e-6;

  for (int i = 0; i < 200; i++) {
    stringstream path_index, pose_index;
    path_index << "image_path" << i;
    pose_index << "pose" << i;
    string image_path;
    Mat pose;
    fs[path_index.str().c_str()] >> image_path;
    fs[pose_index.str().c_str()] >> pose;

    Mat hpose = Mat::eye(4, 4, CV_32F);
    pose(Rect(0, 0, 3, 3)).copyTo(hpose(Rect(0, 0, 3, 3)));
    pose(Rect(3, 0, 1, 3)).copyTo(hpose(Rect(3, 0, 1, 3)));

    Mat img = imread("../../vr_data/" + image_path);
    GaussianBlur(img, img, Size(9,9), 0);
    hmd_data.emplace_back(hmd_frame{img, hpose});
  }

  cv::viz::Viz3d win = cv::viz::getWindowByName("World");
  win.showWidget("Coordinate Widget", WCoordinateSystem());

  CascadeClassifier cascade("../haarcascade_profileface.xml");

  int frame_interval = 2;
  int min_disparity = 0;
  int num_disparities = 32;
  int window_size = 9;
  int smoothness_parameter1 = 8 * hmd_data[0].image.channels() * window_size * window_size;
  int smoothness_parameter2 = 32 * hmd_data[0].image.channels() * window_size * window_size;
  int max_diff = 1;
  int prefilter_cap = 0;
  int uniqueness_ratio = 5;
  int speckle_window_size = 100;
  int speckle_range = 8;
  int sgbm_mode = StereoSGBM::MODE_SGBM;
  //Ptr<cv::StereoBM> bm = cv::StereoBM::create();
  Ptr<StereoSGBM> bm = StereoSGBM::create(min_disparity,
                                          num_disparities, window_size,
                                          smoothness_parameter1,
                                          smoothness_parameter2,
                                          max_diff,
                                          prefilter_cap, uniqueness_ratio, speckle_window_size, speckle_range, sgbm_mode);

  Matx33f e = Matx33f::eye();
  e(0, 0) *= -1;
  e(2, 2) *= -1;

  std::deque<Rect> regions;

  while (!win.wasStopped()) {
    int count = 0;
    cv::MultiTrackerTLD tracker;
    for (int i = 0; i < 200; i++) {
      if (count == 0) {
        vector<Rect> faces;
        cascade.detectMultiScale(hmd_data[i].image, faces, 1.1, 2,
                                 CASCADE_SCALE_IMAGE, Size(60, 60));
        if (faces.size()) {
          // TODO(umar): Ability to track Multiple faces
          tracker.addTarget(hmd_data[i].image, faces[0], "MIL");
        }
        count += faces.size();
      } else {
        tracker.update(hmd_data[i].image);
      }
      Mat cjust_faces = Mat::zeros(hmd_data[i].image.size(), hmd_data[i].image.type());

      for (int f = 0; f < tracker.boundingBoxes.size(); f++) {
        regions.push_back(tracker.boundingBoxes[f]);
        hmd_data[i].image.copyTo(cjust_faces);

        if (i > frame_interval) {
          Mat rjust_faces = Mat::zeros(hmd_data[i-frame_interval].image.size(), hmd_data[i-frame_interval].image.type());
          Mat ljust_faces = Mat::zeros(hmd_data[i-frame_interval].image.size(), hmd_data[i-frame_interval].image.type());
          Mat limg, rimg,
            lpose, rpose;
          Rect rregion, lregion;

          // Figure out which frame is on the left. This is done by analyzing
          // the translation of the HMD and determining how the frame is moving
          auto cpose = hmd_data[i].pose;
          auto ppose = hmd_data[i - frame_interval].pose;
          auto rel_pose_mat = get_relative_pose(ppose, cpose);
          auto[R, t] = get_rotation_translation(rel_pose_mat);
          if (t[0]<0) {
            rregion = regions.back();
            rimg = hmd_data[i].image;
            rpose = hmd_data[i].pose;

            lregion = regions.front();
            limg = hmd_data[i-frame_interval].image;
            lpose = hmd_data[i-frame_interval].pose;
          } else {
            lregion = regions.back();
            limg = hmd_data[i].image;
            lpose = hmd_data[i].pose;

            rregion = regions.front();
            rimg = hmd_data[i-frame_interval].image;
            rpose = hmd_data[i-frame_interval].pose;
          }
          regions.pop_front();

          rel_pose_mat = get_relative_pose(lpose, rpose);
          std::tie(R, t) = get_rotation_translation(rel_pose_mat);

          Mat Rmat, tmat;
          Mat(R).convertTo(Rmat, CV_64F);
          Mat(t).convertTo(tmat, CV_64F);

          // Rectify the images and create a disparity map
          Mat Rl, Rr, Pl, Pr, Q;
          Rect roi1, roi2;
          stereoRectify(intrinsics, Mat(), intrinsics, Mat(), image_size,
                        Rmat, tmat, Rl, Rr, Pl, Pr, Q, 1
                        , -1, Size(), &roi1, &roi2);

          Mat mapl1, mapl2, mapr1, mapr2;
          initUndistortRectifyMap(intrinsics, Mat(), Rl, Pl, image_size, CV_16SC2,
                                  mapl1, mapl2);
          initUndistortRectifyMap(intrinsics, Mat(), Rr, Pr, image_size, CV_16SC2,
                                  mapr1, mapr2);
          //Mat frame, pframe;
          Mat rect_limg, rect_rimg;
          Mat face_mask = Mat::zeros(hmd_data[i-frame_interval].image.size(), CV_8U);
          face_mask(lregion) = 255;

          remap(limg, rect_limg, mapl1, mapl2, cv::INTER_LINEAR);
          remap(rimg, rect_rimg, mapr1, mapr2, cv::INTER_LINEAR);

          cv::line(rect_limg, {0, 90}, {612, 90}, CV_RGB(255, 255, 255));
          cv::line(rect_rimg, {0, 90}, {612, 90}, CV_RGB(255, 255, 255));
          imshow("rect_limg", rect_limg);
          imshow("rect_rimg", rect_rimg);

          Mat disp, disp16;
          Mat img1, img2;
          bm->compute(rect_limg, rect_rimg, disp);
          normalize(disp, disp16, 0, 1<<16, cv::NORM_MINMAX, CV_16U);
          remap(face_mask, face_mask, mapl1, mapl2, cv::INTER_LINEAR);
          disp.copyTo(face_mask, face_mask);
          Mat depth = Mat::zeros(disp.size(), CV_32F);
          float sum = 0;
          int count = 0;
          float baseline = norm(t);
          for (int ii = 0; ii < face_mask.size().height; ii++) {
            for (int jj = 0; jj < face_mask.size().width; jj++) {
              if (int disp = std::abs(face_mask.at<int>(ii, jj))) {
                float& ref = depth.at<float>(ii, jj);
                ref = (baseline * focal_length)/(disp*pixel_size);
                sum += ref;
                count++;
              }
            }
          }
          printf("sum: %f avg: %f\n", sum, sum/count);
          imshow("mask", face_mask);
          imshow("depth", depth);
          imshow("disparity", disp16);

          // imshow("disparity", disp);
          // imshow("disparity16", disp16);
          auto[RR, tt] = get_rotation_translation(hmd_data[i].pose);
          Affine3f rrpose(Mat(RR * e), tt);

          float Z = sum/count;
          auto img_widget = cv::viz::WImage3D(limg, cv::Size2d(Z, Z));
          win.showWidget("face", img_widget);

          Mat face_tt = Mat(RR * e * cv::Vec3f(0.0f, 0.0f, Z)+ tt);

          auto img_pose = Affine3f(RR * e, face_tt);
          win.setWidgetPose("face", img_pose);

          flip(ljust_faces, ljust_faces, -1);
          flip(rjust_faces, rjust_faces, -1);
          auto ccam = WCameraPosition(Matx33d(intrinsics), limg, 0.2);
          win.showWidget("cam", ccam);
          win.setWidgetPose("cam", rrpose);
          win.spinOnce(1);
        }
      }
    }
  }
  // bm_demo(win, hmd_data, intrinsics, image_size);
}
