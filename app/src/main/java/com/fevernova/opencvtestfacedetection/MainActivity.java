package com.fevernova.opencvtestfacedetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.support.annotation.RawRes;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoWriter;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import static org.opencv.core.CvType.CV_8UC3;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getSimpleName();

    private VideoWriter videoWriter;
    private File externalStoragePublicDirectory;
    private Mat rgba;
    private Mat decorationMat;
    private CascadeClassifier eyesClassifier;
    private Bitmap decoration;
    private CascadeClassifier faceClassfier;
    private int absoluteFaceSize = 0;
    private JavaCameraView cameraView;
    private BaseLoaderCallback loaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            super.onManagerConnected(status);
            switch (status) {
                case SUCCESS:
                    initializeOpenCVDependencies();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
        loadOpenCv();
    }



    private static void loadOpenCv() {
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "static initializer: loaded");
        } else {
            Log.d(TAG, "static initializer: not loaded");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_main);

        decoration = BitmapFactory.decodeResource(getResources(), R.drawable.fire);
        decorationMat = bitmapToMat(decoration);
        // Example of a call to a native method
        cameraView = (JavaCameraView) findViewById(R.id.cameraView);
        cameraView.setCvCameraViewListener(this);
        externalStoragePublicDirectory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES);
//        VideoCapture videoCapture = new VideoCapture(externalStoragePublicDirectory.getAbsolutePath());
//        int width = (int) videoCapture.get(CAP_PROP_FRAME_WIDTH);
//        int height = (int) videoCapture.get(CAP_PROP_FRAME_HEIGHT);
//        VideoWriter.fourcc("mp4v");
        videoWriter = new VideoWriter();

    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraView != null) {
            cameraView.disableView();
            videoWriter.release();
        }
    }

    private void initializeOpenCVDependencies() {
        eyesClassifier = initCascadeClassifier(R.raw.eyes_cascade, "eyes_cascade");
        faceClassfier = initCascadeClassifier(R.raw.frontal_cascade_alt, "face_cascade");
        // And we are ready to go
        cameraView.enableView();
        videoWriter.open(externalStoragePublicDirectory.getAbsolutePath() + "/test.avi", VideoWriter.fourcc('M', 'J', 'P', 'G'), 20, new Size(400, 400));
    }

    private CascadeClassifier initCascadeClassifier(@RawRes int res, String fileName) {
        CascadeClassifier cascadeClassifier;
        File cascadeDir = getDir(fileName, Context.MODE_PRIVATE);
        File cascadeFile = new File(cascadeDir, fileName + ".xml");
        try (InputStream is = getResources().openRawResource(res);
             FileOutputStream os = new FileOutputStream(cascadeFile)) {
            byte[] buffer = new byte[4096];
            int bytesRead;

            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        // Load the cascade classifier
        cascadeClassifier = new CascadeClassifier(cascadeFile.getAbsolutePath());
        cascadeClassifier.load(cascadeFile.getAbsolutePath());
        if (cascadeClassifier.empty()) {
            Log.e(TAG, "Failed to load cascade classifier");
            cascadeClassifier = null;
        }

        return cascadeClassifier;
        // And we are ready to go
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "static initializer: loaded");
            loaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Log.d(TAG, "static initializer: not loaded");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_3_0, this, loaderCallback);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        rgba = new Mat(height, width, CV_8UC3);
    }

    @Override
    public void onCameraViewStopped() {
        if (rgba != null)
            rgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        rgba = inputFrame.rgba();
        Core.flip(rgba, rgba, 1);
        Mat resizeImage = new Mat();
        Size sz = new Size(400, 400);
        Imgproc.resize(rgba, resizeImage, sz);

        int height = rgba.rows();
        double factor = 0.5;
        if (Math.round(height * factor) > 0) {
            absoluteFaceSize = (int) Math.round(height * factor);
        }
        
        MatOfRect eyesDetections = new MatOfRect();
        if (eyesClassifier != null)
            eyesClassifier.detectMultiScale(resizeImage, eyesDetections, 1.1, 1, 2, new Size(absoluteFaceSize, absoluteFaceSize), new Size());
//
//        System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));
//        // Draw a bounding box around each face.
//        for (Rect rect : faceDetections.toArray()) {
//            Imgproc.rectangle(rgba, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
//        }
//
        for (Rect rect : eyesDetections.toArray()) {
            Imgproc.circle(resizeImage, new Point(rect.x + rect.width / 2, rect.y + rect.height / 2), 10, new Scalar(0, 255, 0));
        }

        if (videoWriter.isOpened()) {
            videoWriter.write(resizeImage);
        }
        return resizeImage;
    }

    private Mat bitmapToMat(Bitmap bitmap) {
        Mat mat = new Mat();
        Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, mat);
        return mat;
    }
}
