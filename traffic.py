import time, sched
import cv2

import depthai as dai
import numpy as np

#-------------------------------------------------------------------------------
# Pedestrian Traffic Light Detection
#-------------------------------------------------------------------------------

class TrafficPipeline:
    
    def __init__(self, modelPath, labelMap, syncNN = True):

        # Defining variables
        self.modelPath= modelPath
        self.labelMap = labelMap
        self.syncNN = syncNN
        
        # Defining Pipeline
        self.pipeline = dai.Pipeline()
        self.pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)
        
        # Defining Source - 1 Colour Cam + 2 Mono Cam for depth
        camRgb = self.pipeline.createColorCamera()
        camRgb.setPreviewSize(416, 416)
        camRgb.setInterleaved(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        
        monoLeft = self.pipeline.createMonoCamera()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        
        monoRight = self.pipeline.createMonoCamera()
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        # Defining stereoDepth
        stereo = self.pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)
        
        # Creating yolo spatial network node
        spatialDetectionNetwork = self.pipeline.createYoloSpatialDetectionNetwork()
        spatialDetectionNetwork.setBlobPath(self.modelPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(2)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
        spatialDetectionNetwork.setAnchorMasks({ "side26": np.array([1,2,3]), "side13": np.array([3,4,5]) })
        spatialDetectionNetwork.setIouThreshold(0.5)
        
        # Linking the nodes
        camRgb.preview.link(spatialDetectionNetwork.input)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        
        # Output nodes
        xoutRgb = self.pipeline.createXLinkOut()
        xoutNN = self.pipeline.createXLinkOut()
        xoutBoundingBoxDepthMapping = self.pipeline.createXLinkOut()
        xoutDepth = self.pipeline.createXLinkOut()

        xoutRgb.setStreamName("rgb")
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
        xoutDepth.setStreamName("depth")
        
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb.input)
        else:
            camRgb.preview.link(xoutRgb.input)
            
        spatialDetectionNetwork.out.link(xoutNN.input)
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)
        
        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)


    def run(self):
        # Connect and start the pipeline
        with dai.Device(self.pipeline) as device:
        
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
            frame = None
            detections = []
            
            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            red = (0, 0, 255)
            green = (0, 255, 0)
        
            #(new)
            s = sched.scheduler(time.time, time.sleep)
            start_time = time.time()
        
        
            while True:
                inPreview = previewQueue.get()
                inNN = detectionNNQueue.get()
                depth = depthQueue.get()
        
                # (new)
                current_time = time.time()
                elapsed_time = current_time - start_time
                notify_time = int(elapsed_time) % 5
        
                # (new)
                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time
        
                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame()
        
                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                detections = inNN.detections
                if len(detections) != 0:
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()
        
                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)
        
                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
        
        
                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width  = frame.shape[1]
                for detection in detections:
                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = self.labelMap[detection.label]
                    except:
                        label = detection.label
                        
                    if label == 'red':
                        color = red
                    else:
                        color = green
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)/10} cm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)/10} cm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)/10} cm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
                
                msg = None
                if len(detections) == 1:
                    if label == 'red':
                        #msg = 'Red light! Dont Cross Yet'
                        color = red
                    else:
                        #msg = 'Green light! Cross Now'
                        color = green
                    cv2.putText(frame, msg, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    
                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
                cv2.imshow("depth", depthFrameColor)
                cv2.imshow("rgb", frame)
        
                # (new)
                key = cv2.waitKey(1)      
                if key == ord('q'):
                    cv2.destroyAllWindows()                    
                    break