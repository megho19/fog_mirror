package com.example.foggymirror;
 
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
 
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
 
/**
 * MVP fingertip finder: skin-color segmentation -> largest blob -> topmost point.
 *
 * KNOWN LIMITATION: your face is skin too, so this will often lock onto your forehead.
 * It's a stepping stone. The real fix is an ML hand-landmark model (see notes).
 * Keep the interface (Mat in, {x, y} or null out) and you can swap the internals later.
 */
public class FingertipTracker {
 
    // YCrCb skin range (common starting values; tune for your lighting)
    private final Mat lower = new Mat(1, 1, CV_32SC4, new Scalar(0, 133, 77, 0));
    private final Mat upper = new Mat(1, 1, CV_32SC4, new Scalar(255, 173, 127, 0));
    private final Mat kernel = getStructuringElement(MORPH_ELLIPSE, new Size(5, 5));
 
    /** @return {x, y} of the fingertip in frame coordinates, or null if nothing found. */
    public int[] detect(Mat frame) {
        Mat ycrcb = new Mat();
        Mat skin = new Mat();
        cvtColor(frame, ycrcb, COLOR_BGR2YCrCb);
        inRange(ycrcb, lower, upper, skin);
        morphologyEx(skin, skin, MORPH_OPEN, kernel);
 
        MatVector contours = new MatVector();
        findContours(skin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
 
        double bestArea = 0;
        Mat best = null;
        for (long i = 0; i < contours.size(); i++) {
            Mat c = contours.get(i);
            double area = contourArea(c);
            if (area > bestArea) { bestArea = area; best = c; }
        }
 
        int[] result = null;
        if (best != null && bestArea > 3000) {
            IntIndexer idx = best.createIndexer();
            int topX = 0, topY = Integer.MAX_VALUE;
            for (long i = 0; i < idx.size(0); i++) {
                int x = idx.get(i, 0, 0);
                int y = idx.get(i, 0, 1);
                if (y < topY) { topY = y; topX = x; }
            }
            idx.release();
            result = new int[]{topX, topY};
        }
 
        ycrcb.close();
        skin.close();
        return result;
    }
}