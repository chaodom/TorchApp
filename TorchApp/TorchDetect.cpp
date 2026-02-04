#include "TorchDetect.h"
#include "torch/script.h"
#ifndef TORCH_CPU
#include <c10/cuda/CUDAStream.h>
#endif

// 用于分类NMS不同类别的候选框平移量，7680是绝大部分设备的最大分辨率
static const int MAX_WH = 7680;

namespace torchapp {
	using torch::indexing::Slice;
	using torch::indexing::None;

	Mat TorchYolo::letterBox(Mat img)
	{
		Mat padded;
		if (m_scale_fill)
		{
			if (img.size() != m_size)
				cv::resize(img, padded, m_size);
			else
				padded = img.clone();
		}
		else
		{
			// 获取输入图像尺寸
			int img_h = img.rows;
			int img_w = img.cols;
			int target_h = m_size.height;
			int target_w = m_size.width;
			// 计算缩放比例
			float r_h = static_cast<float>(target_h) / img_h;
			float r_w = static_cast<float>(target_w) / img_w;
			float r = std::min(r_h, r_w);
			// 计算缩放后的尺寸
			int new_unpad_w = static_cast<int>(std::round(img_w * r));
			int new_unpad_h = static_cast<int>(std::round(img_h * r));
			// 计算填充量
			int dw = target_w - new_unpad_w;
			int dh = target_h - new_unpad_h;
			// 计算上下左右填充量
			int top = 0, bottom = 0, left = 0, right = 0;
			if (m_isCenter)
			{	// 居中填充（上下/左右均分填充）
				top = static_cast<int>(std::round(dh / 2.0 - 0.1));
				bottom = dh - top;
				left = static_cast<int>(std::round(dw / 2.0 - 0.1));
				right = dw - left;
			}
			else
			{	// 左上角对齐（仅右下填充）
				bottom = dh;
				right = dw;
			}
			// 调整图像大小
			Mat resized;
			if (img_w != new_unpad_w || img_h != new_unpad_h)
				cv::resize(img, resized, cv::Size(new_unpad_w, new_unpad_h));
			else
				resized = img.clone();
			cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar::all(m_padding_value));
		}
		return padded;
	}

	void TorchYolo::scaleBox(cv::Rect& box, Mat img)
	{
		// 获取输入图像尺寸
		int img_h = img.rows;
		int img_w = img.cols;
		int target_h = m_size.height;
		int target_w = m_size.width;
		// 计算缩放比例
		float r_h = static_cast<float>(target_h) / img_h;
		float r_w = static_cast<float>(target_w) / img_w;
		if (m_scale_fill)
		{
			box.x /= r_w;
			box.y /= r_h;
			box.width /= r_w;
			box.height /= r_h;
			box &= cv::Rect(0, 0, img.cols, img.rows);
			return;
		}
		float r = std::min(r_h, r_w);
		if (!m_isCenter)
		{
			box.x /= r;
			box.y /= r;
			box.width /= r;
			box.height /= r;
			box &= cv::Rect(0, 0, img.cols, img.rows);
			return;
		}
		// 计算缩放后的尺寸
		int new_unpad_w = static_cast<int>(std::round(img_w * r));
		int new_unpad_h = static_cast<int>(std::round(img_h * r));
		// 计算填充量
		int dw = target_w - new_unpad_w;
		int dh = target_h - new_unpad_h;
		int pad_x = static_cast<int>(std::round(dw / 2.0 - 0.1));
		int pad_y = static_cast<int>(std::round(dh / 2.0 - 0.1));
		box.x = (box.x - pad_x) / r;
		box.y = (box.y - pad_y) / r;
		box.width /= r;
		box.height /= r;
		box &= cv::Rect(0, 0, img.cols, img.rows);
	}

	Mat TorchYolo::adjustMask(Mat mask, Mat origin_img, const cv::Rect& box)
	{
		Mat mask_scale;
		if (m_scale_fill)
			cv::resize(mask, mask_scale, origin_img.size());
		else
		{
			int img_w = origin_img.cols;
			int img_h = origin_img.rows;
			int mask_w = mask.cols;
			int mask_h = mask.rows;
			float r = std::min(static_cast<float>(mask_h) / img_h, static_cast<float>(mask_w) / img_w);
			int new_unpad_w = static_cast<int>(std::round(img_w * r));
			int new_unpad_h = static_cast<int>(std::round(img_h * r));
			// 计算填充量
			int dw = mask_w - new_unpad_w;
			int dh = mask_h - new_unpad_h;
			// 计算缩放后的尺寸
			int top = m_isCenter ? static_cast<int>(std::round(dh / 2.0 - 0.1)) : 0;
			int bottom = top + new_unpad_h;
			int left = m_isCenter ? static_cast<int>(std::round(dw / 2.0 - 0.1)) : 0;
			int right = left + new_unpad_w;
			if (img_w != new_unpad_w || img_h != new_unpad_h)
				cv::resize(mask.colRange(left, right).rowRange(top, bottom), mask_scale, origin_img.size());
			else
				mask_scale = mask.clone();
		}
		cv::Rect rect = box & cv::Rect(0, 0, mask_scale.cols, mask_scale.rows);
		Mat mask_finel = mask_scale(rect) > 0;
		return mask_finel;
	}

	void TorchYolo::scaleCoord(vector<cv::Point>& points, Mat img)
	{
		// 获取输入图像尺寸
		int img_h = img.rows;
		int img_w = img.cols;
		int target_h = m_size.height;
		int target_w = m_size.width;
		// 计算缩放比例
		float r_h = static_cast<float>(target_h) / img_h;
		float r_w = static_cast<float>(target_w) / img_w;
		if (m_scale_fill)
		{
			for (auto& point : points)
			{
				point.x /= r_w;
				point.y /= r_h;
				point.x = std::clamp(point.x, 0, img.cols);
				point.y = std::clamp(point.y, 0, img.rows);
			}
			return;
		}
		float r = std::min(r_h, r_w);
		if (!m_isCenter)
		{
			for (auto& point : points)
			{
				point.x /= r;
				point.y /= r;
				point.x = std::clamp(point.x, 0, img.cols);
				point.y = std::clamp(point.y, 0, img.rows);
			}
			return;
		}
		// 计算缩放后的尺寸
		int new_unpad_w = static_cast<int>(std::round(img_w * r));
		int new_unpad_h = static_cast<int>(std::round(img_h * r));
		// 计算填充量
		int dw = target_w - new_unpad_w;
		int dh = target_h - new_unpad_h;
		int pad_x = static_cast<int>(std::round(dw / 2.0 - 0.1));
		int pad_y = static_cast<int>(std::round(dh / 2.0 - 0.1));
		for (auto& point : points)
		{
			point.x = (point.x - pad_x) / r;
			point.y = (point.y - pad_y) / r;
			point.x = std::clamp(point.x, 0, img.cols);
			point.y = std::clamp(point.y, 0, img.rows);
		}
	}

	vector<DetectOutput> TorchYolo::detect(Mat img, double scoreThreshold, vector<int> classes)
	{
		return detect(vector<Mat>{ img }, scoreThreshold, classes)[0];
	}

	vector<vector<DetectOutput>> TorchYolo::detect(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		vector<Mat> srcs(imgs.size());
		for (int i = 0; i != imgs.size(); ++i)
			srcs[i] = letterBox(imgs[i]);
		auto result = forward(srcs, true);
		// YOLO11: (batchSize, 4 + num_classes + keypoint_num * keypoint_dimension, num_boxes) num_boxes:(w/32*h/32+w/16*h/16+w/8*h/8)
		// YOLO26: (batchSize, num_boxes, 6 + keypoint_num * keypoint_dimension) num_boxes默认300是最大候选框个数，6是(x1, y1, x2, y2, confidence, class)
		at::Tensor predicts = result.isTensor() ? result.toTensor() : result.toTuple()->elements()[0].toTensor();
		//int channels = predicts.size(1);
		//int num_boxes = predicts.size(2);
		//int num_mask = result.isTensor() ? 0 : result.toTuple()->elements()[1].toTensor().size(1);   // 用于OnnxSegment类调用detect函数
		int batch_size = predicts.size(0);
		vector<vector<DetectOutput>> outputs(batch_size);
		for (int batchIdx = 0; batchIdx != batch_size; ++batchIdx)
		{
			// 提取当前批次的输出
			// YOLO11: (num_classes + 4 + keypoint_num * keypoint_dimension, num_boxes)
			// YOLO26: (num_boxes, 6 + keypoint_num * keypoint_dimension)
			auto predict_batch = predicts[batchIdx];
			outputs[batchIdx] = processOne(&predict_batch, imgs[batchIdx], scoreThreshold, classes);
		}
		return outputs;
	}


	vector<DetectOutput> TorchYolo11::processOne(at::Tensor* pPredict, Mat origin_img, double scoreThreshold, vector<int> classes, at::Tensor* pProto)
	{
		at::Tensor predict = *pPredict;
		int num_class = m_vecClsNames.size();  // 类别数
		int num_boxes = predict.size(1);          // 总框数
		int mi = 4 + num_class;                         // mask系数起始索引
		// -------------------------- 批量筛选候选框 --------------------------
		auto cls_scores = predict.slice(0, 4, mi);  // (nc, num_boxes)
		auto max_scores = cls_scores.amax(0);       // (num_boxes) 每个框的最大类别得分
		auto xc = max_scores > scoreThreshold;         // (num_boxes) 置信度过滤掩码
		// 类别过滤
		auto cls_mask = at::ones_like(xc, at::kBool);
		if (!classes.empty())
		{
			auto cls_ids = cls_scores.argmax(0).to(at::kInt);  // (num_boxes) 最佳类别ID
			cls_mask = at::zeros_like(xc, at::kBool);
			for (int cls : classes)
				cls_mask = cls_mask | (cls_ids == cls);
		}
		// 合并过滤掩码 + 提取保留索引
		auto filt_mask = xc & cls_mask;
		auto keep_indices = filt_mask.nonzero().squeeze();
		if (keep_indices.numel() == 0)
			return {};
		int keep_num = keep_indices.numel();
		// -------------------------- 批量提取过滤后的数据 --------------------------
		// 提取xywh框 (4, keep_num)：cx, cy, w, h
		auto boxes_xywh = predict.slice(0, 0, 4).index({ at::indexing::Ellipsis, keep_indices });
		// 提取得分、类别ID
		auto scores = max_scores.index({ keep_indices });                                         // (keep_num)
		auto cls_ids = cls_scores.argmax(0).index({ keep_indices }).to(at::kInt);                 // (keep_num)
		// 提取mask系数：仅当proto非空时
		at::Tensor mask_coeffs;
		// 提取关键点张量（仅当proto为空且存在关键点维度时）
		at::Tensor keypoints_tensor;
		if (pProto)
			mask_coeffs = predict.slice(0, mi, predict.size(0)).index({ at::indexing::Ellipsis, keep_indices });  // (num_mask, keep_num)
		else if (predict.size(0) > mi)	// keypoints
			keypoints_tensor = predict.slice(0, mi, predict.size(0)).index({ at::indexing::Ellipsis, keep_indices });  // (3*num_kp, keep_num)
		// -------------------------- 批量计算rect坐标 --------------------------
		auto cx = boxes_xywh.slice(0, 0, 1);  // (1, keep_num) 中心x
		auto cy = boxes_xywh.slice(0, 1, 2);  // (1, keep_num) 中心y
		auto w = boxes_xywh.slice(0, 2, 3);   // (1, keep_num) 宽度
		auto h = boxes_xywh.slice(0, 3, 4);   // (1, keep_num) 高度
		auto x = cx - w / 2.0f;
		auto y = cy - h / 2.0f;
		// 类别偏移
		auto c = cls_ids.to(torch::kFloat32) * (m_agnostic ? 0.0f : static_cast<float>(MAX_WH));
		x = x + c.unsqueeze(0);  // (1, keep_num)
		y = y + c.unsqueeze(0);  // (1, keep_num)
		// -------------------------- 转换为OpenCV Rect --------------------------
		std::vector<cv::Rect> boxes_nms;
		std::vector<float> confs_nms;
		// 批量拷贝到CPU
		auto x_cpu = x.to(at::kCPU).contiguous();
		auto y_cpu = y.to(at::kCPU).contiguous();
		auto w_cpu = w.to(at::kCPU).contiguous();
		auto h_cpu = h.to(at::kCPU).contiguous();
		auto scores_cpu = scores.to(at::kCPU).contiguous();
		// 指针遍历（直接读取计算好的x/y/w/h，无额外计算）
		const float* x_ptr = x_cpu.data_ptr<float>();
		const float* y_ptr = y_cpu.data_ptr<float>();
		const float* w_ptr = w_cpu.data_ptr<float>();
		const float* h_ptr = h_cpu.data_ptr<float>();
		const float* s_ptr = scores_cpu.data_ptr<float>();
		for (int i = 0; i < keep_num; ++i)
		{
			float rx = x_ptr[i];  // 左上角x（已加类别偏移）
			float ry = y_ptr[i];  // 左上角y（已加类别偏移）
			float rw = w_ptr[i];  // 宽度
			float rh = h_ptr[i];  // 高度
			boxes_nms.emplace_back(static_cast<int>(rx), static_cast<int>(ry), static_cast<int>(rw), static_cast<int>(rh));
			confs_nms.push_back(s_ptr[i]);
		}
		// -------------------------- NMS  --------------------------
		std::vector<int> nmsIndices;
		cv::dnn::NMSBoxes(boxes_nms, confs_nms, scoreThreshold, m_nmsThreshold, nmsIndices);
		if (nmsIndices.empty())
			return {};
		// -------------------------- 批量计算掩码 --------------------------
		const auto device = predict.device();
		at::Tensor nms_indices_tensor = torch::tensor(nmsIndices, torch::dtype(torch::kLong).device(device));
		cv::Mat masks;
		int proto_h = 0, proto_w = 0;
		if (pProto && !nmsIndices.empty())
		{
			at::Tensor proto = *pProto;
			int num_mask = proto.size(0);
			proto_h = proto.size(1);
			proto_w = proto.size(2);
			// 提取NMS后的mask系数
			auto nms_coeffs = mask_coeffs.index({ at::indexing::Ellipsis, nms_indices_tensor });
			// Proto重塑为二维矩阵
			auto proto_2d = proto.reshape({ num_mask, proto_h * proto_w }).to(torch::kFloat32).to(device);
			// 矩阵乘法
			auto nms_masks = nms_coeffs.t().mm(proto_2d);
			// 转换为OpenCV Mat（修正类型和设备）
			auto nms_masks_cpu = nms_masks.to(torch::kCPU).contiguous();
			masks = cv::Mat(static_cast<int>(nmsIndices.size()), static_cast<int>(proto_h * proto_w), CV_32FC1, nms_masks_cpu.data_ptr<float>()).clone();
		}
		// -------------------- 提取NMS后的关键点张量 --------------------------
		at::Tensor nms_keypoints;
		if (!pProto && keypoints_tensor.defined())
		{
			nms_keypoints = keypoints_tensor.index({ at::indexing::Ellipsis, nms_indices_tensor });  // (3*num_kp, nms_num)
			nms_keypoints = nms_keypoints.t().to(at::kCPU).contiguous();
		}
		// -------------------------- 构造最终输出 --------------------------
		std::vector<DetectOutput> output;
		const float* kp_ptr = nms_keypoints.defined() ? nms_keypoints.data_ptr<float>() : nullptr;
		// 提前拷贝classIds到CPU，后续直接索引
		auto cls_ids_cpu = cls_ids.to(at::kCPU).contiguous();
		const int* cls_ptr = cls_ids_cpu.data_ptr<int>();
		for (size_t i = 0; i != nmsIndices.size(); ++i) 
		{
			int idx = nmsIndices[i];
			if (boxes_nms[idx].width <= 0 || boxes_nms[idx].height <= 0)
				continue;
			DetectOutput output_;
			output_.cls_id = cls_ptr[idx];
			output_.score = confs_nms[idx];
			output_.cls_name = output_.cls_id < m_vecClsNames.size() ? m_vecClsNames[output_.cls_id] : std::to_string(output_.cls_id);
			output_.box = boxes_nms[idx];
			// 移除类别偏移
			const int c = m_agnostic ? 0 : MAX_WH * output_.cls_id;
			output_.box.x -= c;
			output_.box.y -= c;
			// 缩放框到原图尺寸
			scaleBox(output_.box, origin_img);
			if (output_.box.width <= 0 || output_.box.height <= 0)
				continue;
			// 处理mask
			if (pProto)
			{
				auto mask_flat = masks.row(static_cast<int>(i));
				auto mask_2d = mask_flat.reshape(1, proto_h);
				output_.mask = adjustMask(mask_2d, origin_img, output_.box);
			}
			else if (keypoints_tensor.defined())
			{
				// 直接通过Tensor列索引获取当前框的所有关键点（矩阵操作，无循环拷贝）
				const float* curr_kp_ptr = kp_ptr + (i * nms_keypoints.size(1));
				// 初始化关键点容器
				output_.points.resize(nms_keypoints.size(1) / 3);
				// 批量提取x,y坐标（忽略conf值，如需保留可自行添加）
				for (int n = 0; n != output_.points.size(); ++n)
					output_.points[n] = cv::Point(curr_kp_ptr[n * 3], curr_kp_ptr[n * 3 + 1]);
				// 缩放到原图尺寸
				scaleCoord(output_.points, origin_img);
			}
			output.push_back(output_);
		}
		return output;
	}

	vector<DetectOutput> TorchYolo11::detect(Mat img, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo::detect(img, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchYolo11::detect(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo::detect(imgs, scoreThreshold, classes);
	}


	vector<DetectOutput> TorchYolo26::processOne(at::Tensor* pPredict, Mat origin_img, double scoreThreshold, vector<int> classes, at::Tensor* pProto)
	{
		at::Tensor predict = *pPredict;
		// predict维度：(num_boxes, 6 + num_mask)  
		const int mask_start_idx = 6;           // mask系数起始索引（x1,y1,x2,y2,conf,classId,mask...）
		// -------------------------- 批量提取核心数据 --------------------------
		// 提取框坐标 (num_boxes, 4) - x1,y1,x2,y2
		auto boxes_xyxy = predict.slice(1, 0, 4).to(torch::kFloat32);
		// 提取置信度 (num_boxes)
		auto confidences = predict.slice(1, 4, 5).squeeze(1).to(torch::kFloat32);  // 第5列（索引4）
		// 提取类别ID (num_boxes)
		auto cls_ids = predict.slice(1, 5, 6).squeeze(1).to(torch::kInt);  // 第6列（索引5）
		// -------------------------- 批量筛选候选框 --------------------------
		// 置信度过滤掩码：conf > threshold
		auto conf_mask = confidences > scoreThreshold;  // (num_boxes) bool张量
		// 类别过滤掩码
		auto cls_mask = at::ones_like(conf_mask, at::kBool);
		if (!classes.empty()) 
		{
			cls_mask = at::zeros_like(conf_mask, at::kBool);
			for (int cls : classes) 
				cls_mask = cls_mask | (cls_ids == cls);
		}
		// 框有效性掩码：width/height > 0 (x2-x1>0, y2-y1>0)
		auto x1 = boxes_xyxy.slice(1, 0, 1).squeeze(1);
		auto y1 = boxes_xyxy.slice(1, 1, 2).squeeze(1);
		auto x2 = boxes_xyxy.slice(1, 2, 3).squeeze(1);
		auto y2 = boxes_xyxy.slice(1, 3, 4).squeeze(1);
		auto box_valid_mask = ((x2 - x1) > 0) & ((y2 - y1) > 0);  // (num_boxes)
		// 合并所有过滤掩码：置信度达标 + 类别达标 + 框有效
		auto filt_mask = conf_mask & cls_mask & box_valid_mask;  // (num_boxes)
		auto keep_indices = filt_mask.nonzero().squeeze(1);      // 保留的框索引
		if (keep_indices.numel() == 0)
			return {};               // 无符合条件的框，直接返回
		int keep_num = keep_indices.numel();
		// -------------------------- 批量提取过滤后的数据 --------------------------
		// 提取过滤后的框、置信度、类别ID
		auto keep_boxes = boxes_xyxy.index({ keep_indices, at::indexing::Ellipsis });  // (keep_num, 4)
		auto keep_confs = confidences.index({ keep_indices });                         // (keep_num)
		auto keep_cls_ids = cls_ids.index({ keep_indices });                           // (keep_num)
		// 提取mask系数：仅当proto非空时
		at::Tensor mask_coeffs;
		// 提取关键点张量（仅当proto为空且存在关键点维度时）
		at::Tensor keypoints_tensor;
		if (pProto) 
			mask_coeffs = predict.slice(1, mask_start_idx, predict.size(1)).index({ keep_indices, at::indexing::Ellipsis });      // (keep_num, num_mask)
		else if (predict.size(1) > mask_start_idx)	//keypoints
			keypoints_tensor = predict.slice(1, mask_start_idx, predict.size(1)).index({ keep_indices, at::indexing::Ellipsis });  // (keep_num, 3*num_kp)
		
		// -------------------------- 批量拷贝到CPU（仅一次拷贝）-------------------------- 
		auto boxes_cpu = keep_boxes.to(torch::kCPU).contiguous();
		auto confs_cpu_tensor = keep_confs.to(torch::kCPU).contiguous();
		auto cls_ids_cpu_tensor = keep_cls_ids.to(torch::kCPU).contiguous();
		at::Tensor keypoints_cpu;
		if (keypoints_tensor.defined()) 
			keypoints_cpu = keypoints_tensor.to(torch::kCPU).contiguous();  // (keep_num, 3*num_kp)
		// 指针遍历构建Rect
		const float* b_ptr = boxes_cpu.data_ptr<float>();
		const float* c_ptr = confs_cpu_tensor.data_ptr<float>();
		const int* cls_ptr = cls_ids_cpu_tensor.data_ptr<int>();
		const float* kp_ptr = keypoints_cpu.defined() ? keypoints_cpu.data_ptr<float>() : nullptr;
		// -------------------------- 批量计算掩码（对齐Onnx逻辑，无NMS） --------------------------
		const auto device = predict.device();
		cv::Mat masks;
		int proto_h = 0, proto_w = 0;
		if (pProto && keep_num > 0)
		{
			at::Tensor proto = *pProto;
			int num_mask = proto.size(0);
			proto_h = proto.size(1);
			proto_w = proto.size(2);
			// 掩码系数矩阵 (keep_num, num_mask)
			auto mask_coeffs_cpu = mask_coeffs.to(torch::kCPU).contiguous();
			// Proto重塑为二维矩阵 (num_mask, proto_h*proto_w)
			auto proto_2d = proto.reshape({ num_mask, proto_h * proto_w }).to(torch::kFloat32).to(device);
			// 矩阵乘法：(keep_num, num_mask) * (num_mask, h*w) = (keep_num, h*w)
			auto nms_masks = mask_coeffs.mm(proto_2d);
			// 转换为OpenCV Mat
			auto masks_cpu = nms_masks.to(torch::kCPU).contiguous();
			masks = cv::Mat(static_cast<int>(keep_num), static_cast<int>(proto_h * proto_w), CV_32FC1, masks_cpu.data_ptr<float>()).clone();
		}
		// -------------------------- 构造最终输出（对齐Onnx逻辑，无NMS） --------------------------
		std::vector<DetectOutput> output;
		for (size_t i = 0; i != keep_num; ++i)
		{
			float x1 = b_ptr[i * 4 + 0];
			float y1 = b_ptr[i * 4 + 1];
			float x2 = b_ptr[i * 4 + 2];
			float y2 = b_ptr[i * 4 + 3];
			cv::Rect box(x1, y1, x2 - x1, y2 - y1);
			DetectOutput output_;
			output_.cls_id = cls_ptr[i];
			output_.score = c_ptr[i];
			output_.cls_name = output_.cls_id < m_vecClsNames.size() ? m_vecClsNames[output_.cls_id] : std::to_string(output_.cls_id);
			output_.box = box;
			scaleBox(output_.box, origin_img);
			if (output_.box.width <= 0 || output_.box.height <= 0)
				continue;
			// 处理掩码
			if (pProto) 
			{
				cv::Mat mask_flat = masks.row(static_cast<int>(i));
				cv::Mat mask_2d = mask_flat.reshape(1, proto_h);
				output_.mask = adjustMask(mask_2d, origin_img, output_.box);
			}
			// 处理keypoints
			else if (keypoints_tensor.defined())
			{
				// 从批量拷贝的CPU张量中获取当前框的关键点数据（矩阵索引，批量优势）
				const float* curr_kp_ptr = kp_ptr + (i * keypoints_cpu.size(1));
				// 初始化关键点容器
				output_.points.resize(keypoints_cpu.size(1) / 3);
				// 批量提取x,y坐标（忽略conf值，如需保留可自行添加）
				for (int n = 0; n != output_.points.size(); ++n)
					output_.points[n] = cv::Point(curr_kp_ptr[n * 3], curr_kp_ptr[n * 3 + 1]);
				// 缩放到原图尺寸
				scaleCoord(output_.points, origin_img);
			}
			output.push_back(output_);
		}
		return output;
	}

	vector<DetectOutput> TorchYolo26::detect(Mat img, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo::detect(img, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchYolo26::detect(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		return TorchYolo::detect(imgs, scoreThreshold, classes);
	}


	TorchApp::ModelOutput TorchFasterRCNN::forward(Mat img)
	{
		try {
			if (img.channels() != m_iChannel)
				throw std::exception(("expect input channel " + std::to_string(m_iChannel) + " but get " + std::to_string(img.channels())).c_str());
			if (m_mean != cv::Scalar::all(0))
				img -= m_mean;
			if (m_stdev != cv::Scalar::all(1))
				img /= m_stdev;
			auto input = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, torch::kUInt8);
			input = input.to(m_bUseGPU ? torch::kCUDA : torch::kCPU, torch::kFloat).permute({ 2, 0, 1 }).contiguous();
			auto result = m_pModel->forward({ input });
#ifndef TORCH_CPU
			if (m_bUseGPU)
				c10::cuda::getCurrentCUDAStream().synchronize();
#endif
			return result;
		}
		catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
			throw;
		}
	}

	Mat TorchFasterRCNN::resize(Mat img, int min_size, int max_size)
	{
		double scale = min_size * 1.0 / MIN(img.rows, img.cols);
		double newh, neww;
		if (img.rows < img.cols)
		{
			newh = min_size;
			neww = scale * img.cols;
		}
		else
		{
			newh = scale * img.rows;
			neww = min_size;
		}
		if (MAX(newh, neww) > max_size)
		{
			scale = max_size * 1.0 / MAX(newh, neww);
			newh *= scale;
			neww *= scale;
		}
		int w = neww + 0.5;
		int h = newh + 0.5;
		Mat dst;
		cv::resize(img, dst, cv::Size(w, h));
		return dst;
	}

	vector<DetectOutput> TorchFasterRCNN::process(
		at::Tensor* pTensor_clsID,
		at::Tensor* pTensor_score,
		at::Tensor* pTensor_box,
		double scoreThreshold,
		vector<int> classes,
		at::Tensor* pTensor_mask,
		at::Tensor* pTensor_keypoint)
	{
		int num = pTensor_clsID->numel();
		vector<float> scores;
		scores.reserve(pTensor_score->numel());
		vector<cv::Rect2d> rects;
		rects.reserve(num);
		vector<vector<float>> masks;
		masks.reserve(num);
		vector<vector<cv::Point>> keypoints;
		keypoints.reserve(num);
		int mask_w = 0;
		int mask_h = 0;
		if (pTensor_mask)
		{
			mask_w = pTensor_mask->size(2);
			mask_h = pTensor_mask->size(3);
		}
		for (int i = 0; i != num; ++i)
		{
			float score = pTensor_score->data_ptr<float>()[i];
			if (score <= scoreThreshold)
				continue;
			int class_id = pTensor_clsID->data_ptr<int>()[i];
			if (!classes.empty() && std::find(classes.begin(), classes.end(), class_id) == classes.end())
				continue;
			scores.push_back(score);
			float x = pTensor_box->data_ptr<float>()[i * 4] * m_scale_w;
			float y = pTensor_box->data_ptr<float>()[i * 4 + 1] * m_scale_h;
			float w = pTensor_box->data_ptr<float>()[i * 4 + 2] * m_scale_w - x;
			float h = pTensor_box->data_ptr<float>()[i * 4 + 3] * m_scale_h - y;
			rects.push_back(cv::Rect2d(x, y, w, h));
			if (pTensor_mask)
			{
				vector<float> mask;
				masks.emplace_back(pTensor_mask->data_ptr<float>() + i * mask_w * mask_h, pTensor_mask->data_ptr<float>() + (i + 1) * mask_w * mask_h);
			}
			if (pTensor_keypoint)
			{
				int keypoint_num = pTensor_keypoint->size(1);
				vector<cv::Point> points(keypoint_num);
				for (int n = 0; n != keypoint_num; ++n)
				{
					float px = (*pTensor_keypoint)[i].data_ptr<float>()[n * 3] * m_scale_w;
					float py = (*pTensor_keypoint)[i].data_ptr<float>()[n * 3 + 1] * m_scale_h;
					points[n] = cv::Point(px, py);
				}
				keypoints.push_back(points);
			}
		}
		vector<int> nms_result;
		cv::dnn::NMSBoxes(rects, scores, scoreThreshold, m_nmsThreshold, nms_result);
		vector<DetectOutput> outputs(nms_result.size());
		for (int i = 0; i < nms_result.size(); i++)
		{
			DetectOutput output;
			int idx = nms_result[i];
			output.cls_id = pTensor_clsID->data_ptr<int>()[idx];
			output.cls_name = output.cls_id < m_vecClsNames.size() ? m_vecClsNames[output.cls_id] : std::to_string(output.cls_id);
			output.score = scores[idx];
			output.box = rects[idx];
			if (pTensor_mask)
			{
				Mat matMask(mask_w, mask_h, CV_32FC1, masks[idx].data());
				cv::resize(matMask, matMask, cv::Size(output.box.width, output.box.height));
				output.mask = matMask > m_maskThreshold;
			}
			if (pTensor_keypoint)
				output.points = keypoints[idx];
			outputs[i] = output;
		}
		return outputs;
	}

	vector<DetectOutput> TorchFasterRCNN::detect(Mat img, double scoreThreshold, vector<int> classes)
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
		at::Tensor tensor_clsID; 
		at::Tensor tensor_score;
		at::Tensor tensor_box; 
		if (out_tuple.size() == 4)	// detect
		{
			// out_tuple的顺序是(boxes, classes, scores)
			tensor_clsID = out_tuple[1].toTensor().toType(torch::kInt).to(at::kCPU);
			tensor_score = out_tuple[2].toTensor().to(at::kCPU);
			tensor_box = out_tuple[0].toTensor().to(at::kCPU);
		}
		else if (out_tuple.size() == 5)	// segment调用detect方法
		{
			// out_tuple的顺序是(boxes, classes, masks, scores)
			tensor_clsID = out_tuple[1].toTensor().toType(torch::kInt).to(at::kCPU);
			tensor_score = out_tuple[3].toTensor().to(at::kCPU);
			tensor_box = out_tuple[0].toTensor().to(at::kCPU);
		}
		else if (out_tuple.size() == 6)	// keypoint调用detect方法
		{
			// out_tuple的顺序是(boxes, classes, keypoint_heatmaps, keypoints, scores)
			tensor_clsID = out_tuple[1].toTensor().toType(torch::kInt).to(at::kCPU);
			tensor_score = out_tuple[4].toTensor().to(at::kCPU);
			tensor_box = out_tuple[0].toTensor().to(at::kCPU);
		}
		return process(&tensor_clsID, &tensor_score, &tensor_box, scoreThreshold, classes);
	}

	vector<vector<DetectOutput>> TorchFasterRCNN::detect(vector<Mat> imgs, double scoreThreshold, vector<int> classes)
	{
		vector<vector<DetectOutput>> results;
		results.reserve(imgs.size());
		for (auto& img : imgs)
			results.push_back(detect(img, scoreThreshold, classes));
		return results;
	}

}