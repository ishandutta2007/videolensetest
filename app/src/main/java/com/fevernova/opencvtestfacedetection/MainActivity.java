package com.fevernova.opencvtestfacedetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.support.annotation.RawRes;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.AppCompatButton;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.core.CvType.CV_8UC4;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getSimpleName();

    private boolean toApplyLense = false;
    private boolean isRecording = true;
    private AppCompatButton btnApply;
    private AppCompatButton btnRecord;
    private VideoWriter videoWriter;
    private File externalStoragePublicDirectory;
    private Mat rgba;
    private Mat decorationMat;
    private MatOfRect eyesDetections;
    private CascadeClassifier eyesClassifier;
    private Bitmap decoration;
    private CascadeClassifier faceClassfier;
    private int absoluteFaceSize = 0;
    private JavaCameraView cameraView;
    private int maxWidth;
    private int maxHeight;
    private VideoCapture camera;
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

        btnApply = (AppCompatButton) findViewById(R.id.btnApply);
        btnRecord = (AppCompatButton) findViewById(R.id.btnRecord);
        btnApply.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                toApplyLense = !toApplyLense;
            }
        });
        btnRecord.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                isRecording = !isRecording;
            }
        });

        externalStoragePublicDirectory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES);
        camera = new VideoCapture(0);
        decoration = BitmapFactory.decodeResource(getResources(), R.drawable.hypno);
        decorationMat = bitmapToMat(decoration);
        cameraView = (JavaCameraView) findViewById(R.id.cameraView);
        cameraView.setCvCameraViewListener(this);
        maxWidth = 800;
        maxHeight = 600;
        cameraView.setMaxFrameSize(maxWidth, maxHeight);
        videoWriter = new VideoWriter();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraView != null && videoWriter != null && camera != null) {
            cameraView.disableView();
            videoWriter.release();
            camera.release();
        }
    }

    private void initializeOpenCVDependencies() {
        eyesClassifier = initCascadeClassifier(R.raw.eyes_cascade, "eyes_cascade");
        faceClassfier = initCascadeClassifier(R.raw.frontal_cascade_alt, "face_cascade");
        // And we are ready to go
        cameraView.enableView();
        videoWriter.open(externalStoragePublicDirectory.getAbsolutePath() + (System.currentTimeMillis() / 1000) + ".avi",
                VideoWriter.fourcc('M', 'J', 'P', 'G'),
                15,
                new Size(maxWidth, maxHeight),
                true);
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
        Log.d(TAG, "onCameraViewStarted: cameraHeight = " + height + " / cameraWidth = " + width);
    }

    @Override
    public void onCameraViewStopped() {
        if (rgba != null) {
            rgba.release();
        }
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        rgba = inputFrame.rgba();
        int height = rgba.rows();
        //use these to decrease/increase area of face detection -> impacts fps
        double sizeFactor = 0.05;
        int scaleFactor = 2;
        if (Math.round(height * sizeFactor) > 0) {
            absoluteFaceSize = (int) Math.round(height * sizeFactor);
        }

        if (toApplyLense) {
            applyOverlay(scaleFactor);
        }

        //saved frame must match size set in video writer -> resize
        Mat toSaveFrame = new Mat();
        Imgproc.resize(rgba, toSaveFrame, new Size(maxWidth, maxHeight));

        if (videoWriter.isOpened() && isRecording) {
            videoWriter.write(toSaveFrame);
        }

        toSaveFrame.release();
        return rgba;
    }

    private void applyOverlay(int scaleFactor) {
        if (eyesDetections == null) {
            eyesDetections = new MatOfRect();
        }
        //detect eyes in frame
        //fixme shrink the image to improve perfomance?
        if (eyesClassifier != null) {
            eyesClassifier.detectMultiScale(rgba, eyesDetections, scaleFactor, 1, 2,
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        }

        //for each detection -> draw overlay
        for (Rect eyesRect : eyesDetections.toArray()) {
            try {
                // create submatrix where we insert our bitmap
                Mat subMat = rgba.submat(new Rect((int) eyesRect.tl().x,
                        (int) eyesRect.tl().y,
                        decorationMat.cols(),
                        decorationMat.rows()));
                decorationMat.copyTo(subMat);
                subMat.release();
            } catch (CvException e) {
                e.printStackTrace();
            }
        }
    }

    private Mat bitmapToMat(Bitmap bitmap) {
        Mat mat = new Mat(bitmap.getWidth(), bitmap.getHeight(), CV_8UC4, new Scalar(0, 0, 0, 255));
        Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, mat);
        return mat;
    }
}
