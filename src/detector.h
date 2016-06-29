#ifndef DETECTOR_H
#define DETECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "lib/darknet/src/network.h"
#include "lib/darknet/src/parser.h"
#include "lib/darknet/src/utils.h"
#include "lib/darknet/src/detection_layer.h"
#include "lib/darknet/src/box.h"
#include "lib/darknet/src/cost_layer.h"

#ifdef __cplusplus
}
#endif

class Detector
{
public:
    Detector();  
    
    void testExamples(float thresh);

private:
	void testImage(char *inFilename, char *outFilename, float thresh = 0.2f, float nms = 0.5f);
	
	void loadNet(char *cfgfile, char *weightfile);
	
	void loadLabels();
	
    void convertToBoxes(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, int only_objectness = 0);
    
    image _voc_labels[20];
    char const *_voc_names[20];
    
    network _net;
    detection_layer _detLayer;
    box *_boxes;
    float **_probs;
};

#endif // DETECTOR_H
