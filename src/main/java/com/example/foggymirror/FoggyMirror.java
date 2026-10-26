package com.example.foggymirror;
 
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
 
import javax.swing.JFrame;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
 
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
 
/**
 * Foggy mirror: shows a mirrored webcam feed covered in "fog".
 * Wherever the pointer (mouse drag, or fingertip) passes, the fog is wiped away
 * and slowly grows back.
 *
 * Controls: drag mouse to wipe | press H to toggle hand tracking | close window to quit.
 */
public class FoggyMirror {
 
    static volatile int mouseX = -1, mouseY = -1;
 
    public static void main(String[] args) throws Exception {
        OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0); // 0 = default camera
        grabber.setImageWidth(1280);
        grabber.setImageHeight(720);
        grabber.start();
 
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        CanvasFrame window = new CanvasFrame("Foggy Mirror");
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
 
        // Mouse input (step 1: lets you test the fog before hand tracking works)
        window.getCanvas().addMouseMotionListener(new MouseMotionAdapter() {
            @Override public void mouseDragged(MouseEvent e) { mouseX = e.getX(); mouseY = e.getY(); }
        });
        window.getCanvas().addMouseListener(new MouseAdapter() {
            @Override public void mouseReleased(MouseEvent e) { mouseX = -1; mouseY = -1; }
        });
 
        final boolean[] useHand = {false};
        window.addKeyListener(new KeyAdapter() {
            @Override public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_H) useHand[0] = !useHand[0];
            }
        });
 
        FingertipTracker tracker = new FingertipTracker();
        Mat mask = null;   // 0 = fully fogged, 255 = fully clear
        Mat white = null;
 
        while (window.isVisible()) {
            Frame grabbed = grabber.grab();
            if (grabbed == null) break;
 
            Mat raw = converter.convert(grabbed);
            Mat frame = new Mat();
            flip(raw, frame, 1); // mirror horizontally
 
            if (mask == null) {
                mask = new Mat(frame.rows(), frame.cols(), CV_8UC1, Scalar.all(0));
                white = new Mat(frame.size(), frame.type(), Scalar.all(255));
            }
 
            // Fog layer = heavily blurred frame washed out with white
            Mat fog = new Mat();
            GaussianBlur(frame, fog, new Size(51, 51), 0);
            addWeighted(fog, 0.55, white, 0.45, 0, fog);
 
            // Where is the "finger"?
            int[] tip = useHand[0]
                    ? tracker.detect(frame)
                    : (mouseX >= 0 ? new int[]{mouseX, mouseY} : null);
            if (tip != null) {
                circle(mask, new Point(tip[0], tip[1]), 35, Scalar.all(255), -1, LINE_AA, 0);
            }
 
            // Fog slowly re-forms: scale mask down, zero out leftovers
            addWeighted(mask, 0.985, mask, 0, 0, mask);
            threshold(mask, mask, 6, 255, THRESH_TOZERO);
 
            // Soft-edged composite: out = clear * m + fog * (1 - m)
            Mat soft = new Mat();
            GaussianBlur(mask, soft, new Size(21, 21), 0);
            Mat m3 = new Mat();
            cvtColor(soft, m3, COLOR_GRAY2BGR);
            Mat inv = new Mat();
            bitwise_not(m3, inv);
 
            Mat clearPart = new Mat();
            Mat fogPart = new Mat();
            multiply(frame, m3, clearPart, 1.0 / 255.0, -1);
            multiply(fog, inv, fogPart, 1.0 / 255.0, -1);
            Mat out = new Mat();
            add(clearPart, fogPart, out);
 
            window.showImage(converter.convert(out));
 
            // Free native memory promptly (JavaCV Mats are off-heap)
            frame.close(); fog.close(); soft.close(); m3.close();
            inv.close(); clearPart.close(); fogPart.close(); out.close();
        }
 
        grabber.stop();
        window.dispose();
    }
}
 