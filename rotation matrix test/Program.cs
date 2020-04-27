using OpenCvSharp;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;


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

    //----------------------------------------------------------------------------------------------
    /**
     *Calculate distance
     */
    public static double CalculateDistance_R(Vector<double> v1, Vector<double> v2, Matrix<double> R)
    {
        //Calculate the Rotation matrix from A->B: B = R * A
        var transformed_v2 = R * v1;
        //Console.WriteLine("vector :" + transformed_v2);
        var err_vec = transformed_v2 - v2;

        return (err_vec[0] * err_vec[0] + err_vec[1] * err_vec[1] + err_vec[2] * err_vec[2]);
    }

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


    //-------------------------------------------------------------------------------------------------------------------
    /**
     * RANSAC method
     */

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
        Mat srcOri = new Mat("C:/Users/Li&Ao/Desktop/Test/5.JPG", ImreadModes.Grayscale);
        Mat dstOri = new Mat("C:/Users/Li&Ao/Desktop/Test/6.JPG", ImreadModes.Grayscale);

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

        var bestTuple = RansacMethod(betterKp1_tmp, betterKp2_tmp, src.Cols, src.Rows);
        var bestKp1 = bestTuple.Item1;
        var bestKp2 = bestTuple.Item2;

        //Step5:draw matches after ransac
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
       
        Mat outImg = new Mat();
        Cv2.DrawMatches(src, kp1, dst, kp2, plotMatches, outImg);
        Cv2.ImShow("outImg", outImg);
        

        Cv2.Resize(outImg, outImg, new Size(outImg.Rows / 2, outImg.Cols / 2));
        Cv2.ImWrite("C:/Users/Li&Ao/Desktop/Test/output.JPG", outImg);

        //Calculate R matrix    
        Matrix<double> A = Matrix<double>.Build.Dense(3, bestKp1.Count);
        Matrix<double> B = Matrix<double>.Build.Dense(3, bestKp2.Count);
        for (int i = 0; i < bestKp1.Count; i++) {
            Vector<double> p1 = From2dTo3d(bestKp1[i], src.Cols, src.Rows);
            Vector<double> p2 = From2dTo3d(bestKp2[i], src.Cols, src.Rows);
            A.SetColumn(i, p1);
            B.SetColumn(i, p2);
        }
        var R = CalcRotation(A, B);
        Console.WriteLine("R matrix is:" + R);
        Cv2.WaitKey();

    }
}

