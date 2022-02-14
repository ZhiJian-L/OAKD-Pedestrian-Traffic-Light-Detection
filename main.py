from traffic import TrafficPipeline
import cv2

# model to detect pedestrian traffic light
trafficModelPath = r'models/traffic.blob'
trafficLabelMap = ['red', 'green']
trafficPipeline = TrafficPipeline(modelPath = trafficModelPath, labelMap = trafficLabelMap)

try:
    
    trafficPipeline.run()

except:

    cv2.destroyAllWindows()
    raise