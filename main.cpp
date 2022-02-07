#include <iostream>
#include <opencv2/opencv.hpp>



using namespace cv;
using namespace std;

string baseFolder = "/Users/mahtabbigverdi/Documents/Zista/";

double GreenData[9] = {1.0002270e+00,0.0 ,-1.5942768e+01, 0.0 ,9.9994659e-01,-2.0648339e+00, 0.0,0.0 ,1.0000000e+00};

double BlueData[9] = {9.9836475e-01,0.0 ,-7.5636659e+00, 0.0 , 1.0000036e+00 ,-1.5701126e+01,0.0 ,0.0, 1.0000000e+00};

Mat GR_H = Mat(3,3,CV_64F, GreenData);

Mat BR_H = Mat (3, 3, CV_64F, BlueData);


Mat generateCoef(string channelName){
//  options for channelName : line_0_c2_B, line_0_c1_G, line_0_c3_R
    string filename = (baseFolder + channelName).append(".tif");
    Mat img = imread(filename , IMREAD_UNCHANGED);
    int referenceMean = 50000.0;
    Mat_<double> coef;
    for (int i = 0; i < img.cols; i++)
    {
        double meanRow = mean(img.col(i))[0] * 1.0;if (fabs(meanRow - 0) < 0.0000001){
            coef.push_back(1.0);
        }
        else{
            coef.push_back(referenceMean/meanRow);
        }

    }
    return coef.reshape(coef.rows);
}

Mat fixImage(Mat img, Mat coef){
    for (int i = 0; i < img.cols; i++){
        img.col(i) = img.col(i) * coef.at<double>(i) + 0;
    }
    return img;
}



//void calculateDFT(Mat &scr, Mat &dst)
//{
//    // define mat consists of two mat, one for real values and the other for complex values
//    Mat planes[] = { scr, Mat::zeros(scr.size(), CV_64F) };
//    Mat complexImg;
//    merge(planes, 2, complexImg);
//
//    dft(complexImg, complexImg);
//    dst = complexImg;
//}
//
//void fftshift(const Mat &input_img, Mat &output_img)
//{
//    output_img = input_img.clone();
//    int cx = output_img.cols / 2;
//    int cy = output_img.rows / 2;
//    Mat q1(output_img, Rect(0, 0, cx, cy));
//    Mat q2(output_img, Rect(cx, 0, cx, cy));
//    Mat q3(output_img, Rect(0, cy, cx, cy));
//    Mat q4(output_img, Rect(cx, cy, cx, cy));
//
//    Mat temp;
//    q1.copyTo(temp);
//    q4.copyTo(q1);
//    temp.copyTo(q4);
//    q2.copyTo(temp);
//    q3.copyTo(q2);
//    temp.copyTo(q3);
//
//}


int main() {

    Mat R_coef = generateCoef("line_0_c3_R");
    Mat G_coef = generateCoef("line_0_c1_G");
    Mat B_coef = generateCoef("line_0_c2_B");
    Rect cropSize(0, 17968,2048, 4000);
    Mat_<double> RedImage, GreenImage, BlueImage;

//    red
    Mat r_raw = imread("/Users/mahtabbigverdi/Desktop/line_11_c3.tif", IMREAD_GRAYSCALE);
    r_raw = r_raw(cropSize);
    r_raw.convertTo(RedImage, CV_64FC1);
    Mat R_channel = fixImage(RedImage, R_coef);

//    //////////////////////////////
//    Mat padded;                            //expand input image to optimal size
//    int m = getOptimalDFTSize( R_channel.rows );
//    int n = getOptimalDFTSize( R_channel.cols ); // on the border add zero values
//    copyMakeBorder(R_channel, padded, 0, m - R_channel.rows, 0, n - R_channel.cols, BORDER_CONSTANT, Scalar::all(0));
//    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
//    Mat complexI;
//    merge(planes, 2, complexI);
//    dft(complexI, complexI);
//    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
//    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
//    Mat magI = planes[0];
//    magI += Scalar::all(1);                    // switch to logarithmic scale
//    log(magI, magI);
//    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
//    // rearrange the quadrants of Fourier image  so that the origin is at the image center
//    int cx = magI.cols/2;
//    int cy = magI.rows/2;
//    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
//    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
//    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
//    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
//    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
//    q0.copyTo(tmp);
//    q3.copyTo(q0);
//    tmp.copyTo(q3);
//    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
//    q2.copyTo(q1);
//    tmp.copyTo(q2);
//    normalize(magI, magI, 0, 1, NORM_MINMAX);
//    Mat x;
//    magI.convertTo(x, CV_8U);
//    cout << magI.row(0);
//    imshow("spectrum magnitude", magI);
//    waitKey();

//  //////////////////////////////
//
//    Green
    Mat g_raw = imread("/Users/mahtabbigverdi/Desktop/line_11_c1.tif", IMREAD_GRAYSCALE);
    g_raw = g_raw(cropSize);
    g_raw.convertTo(GreenImage, CV_64FC1);
    Mat G_channel = fixImage(GreenImage, G_coef);
//
//  Blue
    Mat b_raw = imread("/Users/mahtabbigverdi/Desktop/line_11_c2.tif", IMREAD_GRAYSCALE);
    b_raw = b_raw(cropSize);
    b_raw.convertTo(BlueImage, CV_64FC1);
    Mat B_channel = fixImage(BlueImage, B_coef);


    Mat fin_R, fin_B, fin_G, fin_img, aligned_G, aligned_B ;
//    convert red channel to unit8
    R_channel.convertTo(fin_R, CV_8UC1);

//    align  green channel and convert to unit8
    warpPerspective (G_channel, aligned_G, GR_H, fin_R.size(), INTER_LINEAR + WARP_INVERSE_MAP);
    aligned_G.convertTo(fin_G, CV_8UC1);

//    align  blue channel and convert to unit8
    warpPerspective (B_channel, aligned_B, BR_H, fin_R.size(), INTER_LINEAR + WARP_INVERSE_MAP);
    aligned_B.convertTo(fin_B, CV_8UC1);

//    stack channels
    vector<Mat> channels;
    channels.push_back(fin_R);
    channels.push_back(fin_G);
    channels.push_back(fin_B);
    merge(channels, fin_img);

    imwrite("/Users/mahtabbigverdi/Desktop/test.png", fin_img);
    return 0;
}
