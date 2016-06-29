#include "detector.h"
#include <fstream>

Detector::Detector() : _voc_names{"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"}
{   
    loadLabels();
    loadNet((char*)"lib/darknet/cfg/yolo.cfg", (char*)"yolo.weights");
}

bool checkFileExists(const char *filename)
{
    std::fstream infile(filename);
    return infile.good();
}

void Detector::loadLabels()
{
    for(int i = 0; i < 20; ++i){
        char buff[256];
        sprintf(buff, "lib/darknet/data/labels/%s.png", _voc_names[i]);
        _voc_labels[i] = load_image_color(buff, 0, 0);
    }
}

void Detector::loadNet(char *cfgfile, char *weightfile)
{
    if (!cfgfile) {
        printf("Choose cfg file!\n");
        return;
    }
    
    if (!weightfile) {
        printf("Choose weights file!\n");
        return;
    }
    
    _net = parse_network_cfg(cfgfile);
    
    load_weights(&_net, weightfile);
    
    _detLayer = _net.layers[_net.n-1];
    
    set_batch_network(&_net, 1);
    
    srand(2222222);
    
    _boxes = (box*)calloc(_detLayer.side*_detLayer.side*_detLayer.n, sizeof(box));
    _probs = (float**)calloc(_detLayer.side*_detLayer.side*_detLayer.n, sizeof(float*));
    
    for(int j = 0; j < _detLayer.side*_detLayer.side*_detLayer.n; ++j) 
		_probs[j] = (float*)calloc(_detLayer.classes, sizeof(float*));
}

void Detector::testExamples(float thresh)
{
    for (int i = 0; i < 50; i++) {
		char inFilename[50];
		char outFilename[50];
		
		sprintf(inFilename, "input/example_%d_1.jpeg", i);
		sprintf(outFilename, "output/example_%d_1", i);
		
		if (!checkFileExists(inFilename)) {
	   		sprintf(inFilename, "input/example_%d_1.jpg", i);
	   		sprintf(outFilename, "output/example_%d_1", i);
	   		
	   		if (!checkFileExists(inFilename)) continue;
		}
		
		testImage(inFilename, outFilename, thresh);
    }
}

void Detector::testImage(char *inFilename, char *outFilename, float thresh, float nms)
{
    image im = load_image_color(inFilename, 0, 0);
    
    printf("%s loaded. Prediction started.\n", inFilename);
    
    image sized = resize_image(im, _net.w, _net.h);
        
    float *X = sized.data;
        
    clock_t time = clock();
        
    float *predictions = network_predict(_net, X);
        
    printf("%s: Predicted in %f seconds.\n", inFilename, sec(clock()-time));
        
    convertToBoxes(predictions, _detLayer.classes, _detLayer.n, _detLayer.sqrt, _detLayer.side, 1, 1, thresh);
        
    if (nms) do_nms_sort(_boxes, _probs, _detLayer.side*_detLayer.side*_detLayer.n, _detLayer.classes, nms);

    draw_detections(im, _detLayer.side*_detLayer.side*_detLayer.n, thresh, _boxes, _probs, (char**)_voc_names, _voc_labels, 20);
	
    save_image(im, outFilename);
        
    free_image(im);
    free_image(sized);
}

void Detector::convertToBoxes(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, int only_objectness)
{
    for (int i = 0; i < side*side; ++i) {
        int row = i / side;
        int col = i % side;
        
        for(int n = 0; n < num; ++n) {
            int index = i*num + n;
            
            int p_index = side*side*classes + i*num + n;
            
            float scale = predictions[p_index];
            
            int box_index = side*side*(classes + num) + (i*num + n)*4;
            
            _boxes[index].x = (predictions[box_index + 0] + col) / side * w;
            _boxes[index].y = (predictions[box_index + 1] + row) / side * h;
            _boxes[index].w = pow(predictions[box_index + 2], (square ? 2 : 1)) * w;
            _boxes[index].h = pow(predictions[box_index + 3], (square ? 2 : 1)) * h;
            
            for(int j = 0; j < classes; ++j) {
                int class_index = i*classes;
                float prob = scale*predictions[class_index+j];
                _probs[index][j] = (prob > thresh) ? prob : 0;
            }
            
            if(only_objectness) {
                _probs[index][0] = scale;
            }
        }
    }
}
