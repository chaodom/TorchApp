#include "TorchSegment.h"
#include <torch/script.h>

namespace torchapp {

	vector<DetectOutput> TorchYoloSeg::segment(Mat img, double scoreThreshold, vector<int> classes)
	{
		return segment(vector<Mat>{img}, scoreThreshold, classes)[0];
	}

	vector<vector<DetectOutput>> TorchYoloSeg::segment(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		vector<Mat> srcs(imgs.size());
		for (int i = 0; i != imgs.size(); ++i)
			srcs[i] = letterBox(imgs[i]);
		auto result = forward(srcs, true).toTuple()->elements();
		// YOLO11: (batchSize, 4 + num_classes+ num_masks, num_boxes) num_boxes:(w/32*h/32+w/16*h/16+w/8*h/8)
		// YOLO26: // (batchSize, num_boxes, 6 + num_masks) num_boxes默认300是最大候选框个数，6是(x1, y1, x2, y2, confidence, class)
		auto predicts = result[0].toTensor();   
		auto protos = result[1].toTensor(); // (batchSize, num_masks,  proto_h, proto_w)

		// 打印
		/*at::Tensor predicts_to_print = predicts[0].index({
		at::indexing::Slice(0, 10),
		at::indexing::Slice(0, 6) });
		at::Tensor protos_to_print = protos[0].index({
		at::indexing::Slice(),
		at::indexing::Slice(0, 1),
		at::indexing::Slice(0, 1) });
		print(predicts_to_print);
		print(protos_to_print);*/

		/*int num_mask = protos.size(1);
		int proto_h = protos.size(2);
		int proto_w = protos.size(3);
		int channels = predicts.size(1);
		int num_boxes = predicts.size(2);*/
		int batch_size = predicts.size(0);
		vector<vector<DetectOutput>> outputs(batch_size);
		for (int batchIdx = 0; batchIdx != batch_size; ++batchIdx)
		{
			// 提取当前批次的输出 
			// YOLO11: (4 + num_classes + num_masks, num_boxes)
			// YOLO26: (num_boxes, 6 + num_masks)
			auto predict_batch = predicts[batchIdx];
			auto proto_batch = protos[batchIdx];
			outputs[batchIdx] = processOne(&predict_batch, imgs[batchIdx], scoreThreshold, classes, &proto_batch);
		}
		return outputs;
	}


	vector<DetectOutput> TorchYolo11Seg::segment(Mat img, double scoreThreshold, vector<int> classes)
	{
		return TorchYoloSeg::segment(img, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchYolo11Seg::segment(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchYoloSeg::segment(imgs, scoreThreshold, classes);
	}


	vector<DetectOutput> TorchYolo26Seg::segment(Mat img, double scoreThreshold, vector<int> classes)
	{
		return TorchYoloSeg::segment(img, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchYolo26Seg::segment(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchYoloSeg::segment(imgs, scoreThreshold, classes);
	}


	vector<DetectOutput> TorchMaskRCNN::segment(Mat img, double scoreThreshold, vector<int> classes)
	{
		ModelOutput result;
		m_scale_w = 1.0;
		m_scale_h = 1.0;
		if (m_bResize)
		{
			Mat img_ = resize(img, m_minSize, m_maxSize);
			result = forward(img_);
			m_scale_w = img.cols * 1.0 / img_.cols;
			m_scale_h = img.rows * 1.0 / img_.rows;
		}
		else
			result = forward(img);
		auto out_tuple = result.toTuple()->elements();
		// out_tuple的顺序是(boxes, classes, masks, scores)
		auto tensor_clsID = out_tuple[1].toTensor().toType(torch::kInt).to(at::kCPU);
		auto tensor_score = out_tuple[3].toTensor().to(at::kCPU);
		auto tensor_box = out_tuple[0].toTensor().to(at::kCPU);
		auto tensor_mask = out_tuple[2].toTensor().to(at::kCPU);
		return process(&tensor_clsID, &tensor_score, &tensor_box, scoreThreshold, classes, &tensor_mask);
	}

	vector<vector<DetectOutput>> TorchMaskRCNN::segment(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		vector<vector<DetectOutput>> results;
		results.reserve(imgs.size());
		for (auto& img : imgs)
			results.push_back(segment(img, scoreThreshold, classes));
		return results;
	}

}