# Camera recognition method for handwritten fonts

实现步骤：

1.正确配置android stdio工程，移植tensorflow和opencv

2. D:\文档\摄像头识别手写字体  目录下已经准备好opencv调用摄像头文件的源码和布局文件，替换工程中的Main文件和xml文件

3.在androidManifest.xml文件中加入摄像头权限

<uses-permission android:name="android.permission.CAMERA" />
<uses-feature android:name="android.hardware.camera" android:required="false"/>
<uses-feature android:name="android.hardware.camera.autofocus" android:required="false"/>
<uses-feature android:name="android.hardware.camera.front" android:required="false"/>
<uses-feature android:name="android.hardware.camera.front.autofocus" android:required="false"/>


4.摄像头权限动态申请

if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
    //申请权限
    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
}


5.如遇手机显示问题以及黑屏闪退问题，需要手动调整分辨率

mOpenCvCameraView.setMaxFrameSize(640, 640);



把上一步的手写字体代码识别复制过来


private TensorFlowInferenceInterface tensorFlowInferenceInterface = null;
private static final String mode_file = "file:///android_asset/MnistTF_model.pb";
private static final String INPUT_NODE = "conv2d_1_input_2:0";
private static final String OUTPUT_NODE = "dense_3_2/Softmax:0";
private float[] inputs_data = new float[784];
private float[] outputs_data = new float[10];


实例Tensorflow接口
tensorFlowInferenceInterface = new TensorFlowInferenceInterface(getAssets(), mode_file);


写一个函数用来识别手写字体

int width = bitmap_roi.getWidth();
int height = bitmap_roi.getHeight();
int[] pixels = new int[width * height];

Log.d("tag", width+"  "+height);

try {
    bitmap_roi.getPixels(pixels, 0, width, 0, 0, width, height);
    for (int i = 0; i < pixels.length; i++) {
        inputs_data[i] = (float)pixels[i];
    }
}catch (Exception e){
    Log.d("tag", e.getMessage());
}

Log.d("Tag", "width: "+width+"   height:"+height);

Trace.beginSection("feed");
tensorFlowInferenceInterface.feed(INPUT_NODE, inputs_data, 1,28,28,1);
Trace.endSection();

Trace.beginSection("run");
tensorFlowInferenceInterface.run(new String[]{OUTPUT_NODE});
Trace.endSection();

Trace.beginSection("fetch");
tensorFlowInferenceInterface.fetch(OUTPUT_NODE, outputs_data);
Trace.endSection();

int logit = 0;
for(int i=1;i<10;i++)
{
    if(outputs_data[i]>outputs_data[logit])
        logit=i;
}

if(outputs_data[logit]>0)
    return logit;
return -1;


我们的环境是白底黑字，接下来我们要用opencv把我们写的字从图片中分割出来

Mat img_t = new Mat();
Mat img_contours;

if(img_rgb != null){
    Imgproc.cvtColor(img_rgb, img_gray, Imgproc.COLOR_RGB2GRAY);

    Imgproc.threshold(img_gray, img_gray,140,255,Imgproc.THRESH_BINARY_INV);
    Mat ele1=Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3));
    Mat ele2=Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(6,6));
    Imgproc.erode(img_gray,img_gray,ele1);
    Imgproc.dilate(img_gray,img_gray,ele2);

    img_contours = img_gray.clone();
    List<MatOfPoint> contours=new ArrayList<>();
    Imgproc.findContours(img_contours,contours,new Mat(),
            Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_SIMPLE);
    for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++)
    {
        double contourArea = Imgproc.contourArea(contours.get(contourIdx));
        Rect rect =  Imgproc.boundingRect(contours.get(contourIdx));
        if(contourArea<1500||contourArea>20000)
            continue;

        Mat roi = new Mat(img_gray, rect);
        Imgproc.resize(roi,roi,new Size(28,28));

        Bitmap bitmap2 = Bitmap.createBitmap(roi.width(),roi.height(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(roi,bitmap2);
        int number = toNumber(bitmap2);
        if(number>=0) {
            //tl左上角顶点  br右下角定点
            double x = rect.tl().x;
            double y = rect.br().y;
            Point p = new Point(x,y);
            Imgproc.rectangle(img_rgb,rect.tl(),rect.br(), new Scalar(0,0,255));
            Imgproc.putText(img_rgb, Integer.toString(number), p, Core.FONT_HERSHEY_DUPLEX,
                    6, new Scalar(0,0,255), 2);
        }
    }


