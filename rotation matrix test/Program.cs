using OpenCvSharp;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

//--------------------------------------------------------------------------------------------------

class calcR
{

    //-------------------------------------------------------------------------------------------------
    /**
     * Calculate the Rotation matrix from A->B: B = R*A
     */
    public static Matrix<double> CalcRotation(Matrix<double> A, Matrix<double> B)
    {
        var M = Matrix<double>.Build;
        var H = A * B.Transpose();
        var svd = H.Svd();
        var tmp = svd.U * svd.VT;//add() or not???
        var e = M.DenseOfDiagonalArray(new[] { 1, 1, tmp.Determinant() });
        var R = svd.VT.Transpose() * e * svd.U.Transpose();
        return R;
    }

    //--------------------------------------------------------------------------------------------------

    /*
     * Homography: Calculate the H matirx.
     * H = A-1*B
     */
    public static Matrix<double> FindHomography(List<Point2d> points_src, List<Point2d> points_dst)
    {
        // Check if the two points are of same size and the size the larger than 4
        Debug.Assert(points_src.Count == points_dst.Count);
        Debug.Assert(points_src.Count >= 4);

        // Ax = B, x is the array of items in homography matrix.
        var A = Matrix<double>.Build.Dense(2 * points_src.Count, 8);
        for (int i = 0; i < points_src.Count; i += 2)
        {
            double u = points_dst[i].X;
            double v = points_dst[i].Y;
            double x = points_src[i].X;
            double y = points_src[i].Y;
            double[,] row1 = new double[1, 8] { { -x, -y, -1, 0, 0, 0, u * x, u * y } };
            var row_1 = Matrix<double>.Build.DenseOfArray(row1);
            double[,] row2 = new double[1, 8] { { 0, 0, 0, -x, -y, -1, v * x, v * y } };
            var row_2 = Matrix<double>.Build.DenseOfArray(row2);
            A.SetSubMatrix(i, 0, row_1);
            A.SetSubMatrix(i + 1, 0, row_2);
        }

        var B = Matrix<double>.Build.Dense(2 * points_src.Count, 1);
        for (int i = 0; i < points_src.Count; i += 2)
        {
            B[i, 0] = -1 * points_dst[i].X;
            B[i + 1, 0] = -1 * points_dst[i].Y;
        }

        // Solve Ax = B with least square
        var h = A.Solve(B);

        var H = Matrix<double>.Build.Dense(3, 3);
        H[0, 0] = h[0, 0]; H[0, 1] = h[1, 0]; H[0, 2] = h[2, 0];
        H[1, 0] = h[3, 0]; H[1, 1] = h[4, 0]; H[1, 2] = h[5, 0];
        H[2, 0] = h[6, 0]; H[2, 1] = h[7, 0]; H[2, 2] = 1;

        return H;
    }

    //-for Homograpgy--------------------------------------------------------------------------------------------------------------

    public static double CalculateDistance_H(Point2d kp1, Point2d kp2, Matrix<double> H)
    {
        var c1 = Vector<double>.Build.DenseOfArray(new[] { kp1.X, kp1.Y, 1 });
        var c2 = Vector<double>.Build.DenseOfArray(new[] { kp2.X, kp2.Y, 1 });

        var transformed_c1 = H * c1;
        var err_vec = transformed_c1 - c2;

        return (err_vec[0] * err_vec[0] + err_vec[1] * err_vec[1]);
    }

    //--for Rotation matrix--------------------------------------------------------------------------------------------

    public static double CalculateDistance_R(Vector<double> v1, Vector<double> v2, Matrix<double> R)
    {
        //Calculate the Rotation matrix from A->B: B = R * A
        var transformed_v2 = R * v1;
        //Console.WriteLine("vector :" + transformed_v2);
        var err_vec = transformed_v2 - v2;

        return (err_vec[0] * err_vec[0] + err_vec[1] * err_vec[1] + err_vec[2] * err_vec[2]);
    }

    //public static double CalculateDistance_R(Point2d kp2, Vector<double> v1, Matrix<double> R, int W, int H)
    //{
    //    //Calculate the Rotation matrix from A->B: B = R * A
    //    var tran_v2 = R * v1;
    //    Console.WriteLine("vector :" + tran_v2);
    //    double x = tran_v2[0];
    //    double y = tran_v2[1];
    //    double z = tran_v2[2];

    //    //inverse 3D tp 2D point
    //    double tran_alpha = Math.Atan(z / x);
    //    double tran_beta = Math.Asin(y / Math.Sqrt(x*x + y*y + z*z));

    //    double tran_u = (tran_alpha / 2 * Math.PI) + 0.5;
    //    double tran_v = 0.5 - (tran_beta / Math.PI);

    //    double tran_Y = tran_u * W - 0.5;
    //    double tran_X = tran_v * H - 0.5;
    //    Console.WriteLine("tran_x :" + tran_X);
    //    Console.WriteLine("tran_y:" + tran_Y);
    //    double error_X = kp2.X - tran_X;
    //    double error_Y = kp2.Y - tran_Y;
    //    Console.WriteLine("kp2.x :" + kp2.X);
    //    Console.WriteLine("kp2.y :" + kp2.Y);
    //    double error = Math.Sqrt( error_X * error_X + error_Y * error_Y);
    //    return error;
    //}
    //--------------------------------------------------------------------------------------------------

    /**
     * Project from 2d image coordinate to 3d camera coordinate
     */
    public static Vector<double> From2dTo3d(Point2d p, int W, int H)
    {
        
        double m = p.Y;
        double n = p.X;
        double u = (m + 0.5) / W;
        double v = (n + 0.5) / H;
        double alpha = (u - 0.5) * 2 * Math.PI;
        double beta = (0.5 - v) * Math.PI;
        
        double X = Math.Cos(beta) * Math.Cos(alpha);
        double Y = Math.Sin(beta);
        double Z = Math.Cos(beta) * Math.Sin(alpha);
        var V = Vector<double>.Build;
        var ret = V.DenseOfArray(new[] { X, Y, Z });
        return ret;
    }


    //-----------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------------------

    public static Tuple<List<Point2d>, List<Point2d>> RansacMethod(List<Point2d> kp1, List<Point2d> kp2, int W, int H)
    {
        var betterkp1 = new List<Point2d>();
        var betterkp2 = new List<Point2d>();
        var betterkpTuple = new Tuple<List<Point2d>, List<Point2d>>(betterkp1, betterkp2);

        double error_thred = 0.7f;
        int num = 4;
        int maxNumberOfInlier = num;
        double max_iter = 1000.0f;
        double w = 0.70f;
        double p = 0.99f;

        //iteration
        int iteration = 0;
        while (iteration < max_iter)
        {
            var maybeInlier1 = new List<Point2d>();
            var maybeInlier2 = new List<Point2d>();
            var alsoInlier1 = new List<Point2d>();
            var alsoInlier2 = new List<Point2d>();

            //random select four pair of points
            
            var IndexOfMaybeInlier = new List<int>();
            while (maybeInlier1.Count < num)
            {
                var random = new Random();
                int index = random.Next(kp1.Count);
                if (!maybeInlier1.Contains(kp1[index]))
                {
                    maybeInlier1.Add(kp1[index]);
                    maybeInlier2.Add(kp2[index]);
                    IndexOfMaybeInlier.Add(index);
                }
                else
                    continue;
            }

            //calculate model(R matrix) by using the 4 random select points
            //transform 2d to 3d codinate(x,y,z)
            Matrix<double> A = Matrix<double>.Build.Dense(3, num);
            Matrix<double> B = Matrix<double>.Build.Dense(3, num);
            for (int i = 0; i < num; i++)
            {
                Vector<double> p1 = From2dTo3d(maybeInlier1[i], W, H);
                Vector<double> p2 = From2dTo3d(maybeInlier2[i], W, H);
                A.SetColumn(i, p1);
                B.SetColumn(i, p2);
            }
            Matrix<double> R = CalcRotation(A, B);

            //calculate the rest data, if it fits the model, add it to alsoInlier.
            for (int i = 0; i < kp1.Count; i++)
            {
                if (IndexOfMaybeInlier.Contains(i))
                    continue;

                //Point2d to (x,y,z)
                Vector<double> v1 = From2dTo3d(kp1[i], W, H);
                Vector<double> v2 = From2dTo3d(kp2[i], W, H);
                double err = CalculateDistance_R(v1, v1, R);
                Console.WriteLine("err is " + err);
                if (err < error_thred)
                {
                    alsoInlier1.Add(kp1[i]);
                    alsoInlier2.Add(kp2[i]);
                }
            }


            // end if We found enough number of inlier.
            var numberOfInlier = alsoInlier1.Count + maybeInlier1.Count;
            if (numberOfInlier >= maxNumberOfInlier)
            {
                maxNumberOfInlier = numberOfInlier;
                betterkp1.Clear();
                betterkp2.Clear();
                betterkp1.AddRange(maybeInlier1);
                betterkp2.AddRange(maybeInlier2);
                betterkp1.AddRange(alsoInlier1);
                betterkp2.AddRange(alsoInlier2);
            }

            // k is dymatic.
            double inlierProb = maxNumberOfInlier / kp1.Count;
            if (inlierProb > w)
                w = inlierProb;
            max_iter = Math.Log(1 - p) / Math.Log(1 - Math.Pow(w, num));
            iteration++;

        }

        return betterkpTuple;
    }

    //-----------------------------------------------------------------------------------------------------

    /*
     *Rotate images 90 degrees
     */
    public static void RotateAndResize(Mat src, out Mat dst, bool isRight/*Left is Basic*/)
    {
        dst = new Mat();
        var center = new Point2f(src.Cols / 2, src.Cols / 2);

        Mat rotationMat = Cv2.GetRotationMatrix2D(center, 90, 1);
        Cv2.WarpAffine(src, dst, rotationMat, new Size(src.Rows, src.Cols));

        if (isRight)
        {
            Cv2.Flip(dst, dst, FlipMode.XY);
        }
    }


    //--------------------------------------------------------------------------------------------------


    static void Main()
    {
        Mat srcOri = new Mat("C:/Users/Li&Ao/Desktop/Test/3.JPG", ImreadModes.Grayscale);
        Mat dstOri = new Mat("C:/Users/Li&Ao/Desktop/Test/4.JPG", ImreadModes.Grayscale);

        Mat src = new Mat();
        Mat dst = new Mat();
        RotateAndResize(srcOri, out src, true);
        RotateAndResize(dstOri, out dst, true);


        // Step1: Detect the keypoints and generate their descriptors using SURF
        ORB orb = ORB.Create();
        KeyPoint[] kp1, kp2;
        Mat desc1 = new Mat();
        Mat desc2 = new Mat();
        orb.DetectAndCompute(src, null, out kp1, desc1);
        orb.DetectAndCompute(dst, null, out kp2, desc2);

        // Step2: Matching descriptor vectors with a brute force matcher
        var bfMatcher = new BFMatcher();
        var matches = bfMatcher.KnnMatch(desc1, desc2, k: 2);

        // Step3: Ratio test for outlier removal
        var betterKp1 = new List<Point2f>();
        var betterKp2 = new List<Point2f>();
        var betterMatches = new List<DMatch>();
        foreach (DMatch[] items in matches)
        {
            if (items[0].Distance < 0.8 * items[1].Distance)
            {
                betterKp1.Add(kp1[items[0].QueryIdx].Pt);
                betterKp2.Add(kp2[items[0].TrainIdx].Pt);
                betterMatches.Add(items[0]);
            }
        }

        // Step4: RANSAC for outlier removal
        Point2d Point2fToPoint2d(Point2f pf) => new Point2d(((double)pf.X), ((double)pf.Y));
        var betterKp1_tmp = betterKp1.ConvertAll(Point2fToPoint2d);
        var betterKp2_tmp = betterKp2.ConvertAll(Point2fToPoint2d);
        var output = new Mat();

        // useRANSAC to calculate
        var bestTuple = RansacMethod(betterKp1_tmp, betterKp2_tmp, src.Cols, src.Rows);
        var bestKp1 = bestTuple.Item1;
        var bestKp2 = bestTuple.Item2;

        // draw matches after ransac
        var plotMatches = new List<DMatch>();
        foreach (DMatch[] items in matches)
        {
            var p1 = Point2fToPoint2d(kp1[items[0].QueryIdx].Pt);
            var p2 = Point2fToPoint2d(kp2[items[0].TrainIdx].Pt);
            if (bestKp1.Contains(p1) && bestKp2.Contains(p2))
            {
                plotMatches.Add(items[0]);
            }
        }
        Console.WriteLine("betterkp size is " + betterKp1.Count);
        Console.WriteLine("match size is " + plotMatches.Count);
        Console.WriteLine("kp size is " + bestKp2.Count);
        Mat outImg = new Mat();
        Cv2.DrawMatches(src, kp1, dst, kp2, plotMatches, outImg);
        Cv2.ImShow("outImg", outImg);
        

        Cv2.Resize(outImg, outImg, new Size(outImg.Rows / 2, outImg.Cols / 2));
        Cv2.ImWrite("C:/Users/Li&Ao/Desktop/Test/output-5-6-4.JPG", outImg);

        // test R matrix    
        //Matrix<double> A = Matrix<double>.Build.Dense(3, bestKp1.Count);
        //Matrix<double> B = Matrix<double>.Build.Dense(3, bestKp2.Count);
        //for (int i = 0; i < bestKp1.Count; i++)
        //{
        //    Vector<double> p1 = From2dTo3d(bestKp1[i], src.Cols, src.Rows);
        //    Vector<double> p2 = From2dTo3d(bestKp2[i], src.Cols, src.Rows);
        //    A.SetColumn(i, p1);
        //    B.SetColumn(i, p2);
        //}
        //var R = CalcRotation(A, B);
        //Console.WriteLine("R matrix is:" + R);
        //Cv2.WaitKey();
        //-----------------------------------------------------------------------------------
        Matrix<double> A = Matrix<double>.Build.Dense(3, 3);
        Matrix<double> B = Matrix<double>.Build.Dense(3, 3);
        for (int i = 0; i < 3; i++)
        {
            Vector<double> p1 = From2dTo3d(bestKp1[i], src.Cols, src.Rows);
            Vector<double> p2 = From2dTo3d(bestKp2[i], src.Cols, src.Rows);
            A.SetColumn(i, p1);
            B.SetColumn(i, p2);
        }
        var R = CalcRotation(A, B);
        Console.WriteLine("R matrix is:" + R);
        Cv2.WaitKey();


        // test homography
        //Mat H_mat = new Mat(new Size(3, 3), MatType.CV_64FC1);
        //for (int i = 0; i < 3; i++)
        //{
        //    for (int j = 0; j < 3; j++)
        //    {
        //        H_mat.Set<double>(i, j, R[i, j]);
        //    }
        //}
        //Cv2.WarpPerspective(src, src, H_mat, src.Size());

        // plot after homography
        //Mat plot_img = new Mat(new Size(src.Width, src.Height + dst.Height), MatType.CV_8UC3);
        //plot_img.SetTo(new Scalar(0, 0, 0));
        //var tmp1 = new Mat(plot_img, new Rect(0, 0, src.Width, src.Height));
        //var tmp2 = new Mat(plot_img, new Rect(0, src.Height - 1, dst.Width, dst.Height));
        //src.CopyTo(tmp1);
        //dst.CopyTo(tmp2);
        //Cv2.ImShow("plot", plot_img);
        //Cv2.WaitKey();

    }
}

