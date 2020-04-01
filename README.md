To run the triangulation part run Warpper_triangulation.py and make sure it is in the same directory as Data folder
You must have the predictor file in the directory or you can specify the directory at line 158 for ex. [ (predictor = dlib.shape_predictor("info.dat")) ]
You can manually change path of source face (image) at line 160, for ex. [ img_src = cv2.imread('Data/Scarlett.jpg') ] and
Target face (video) at line 167, for ex. [ out = cv2.VideoWriter('Test3OutputTri.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 6, (frame_width,frame_height)) ]
Change the path for Output directory ('Test3OutputTri.avi') at line 167 for ex. [ out = cv2.VideoWriter('Test3OutputTri.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 6, (frame_width,frame_height)) ]

To run the TPS part, just run the Warpper.py, which will load the 3 videos and generate the output video. You can modify the input and output path in the main function at Line 507.