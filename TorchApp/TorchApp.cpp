#include "TorchApp.h"
#include <fstream>
#include <torch/script.h>
#include <torchvision/vision.h>
#ifndef TORCH_CPU
#include <c10/cuda/CUDAStream.h>
#endif

using std::ifstream;
using std::cerr;
using std::endl;
using std::exception;
using std::pair;

namespace torchapp {

	void TorchApp::initial(const string& model_file, cv::Size size, const string& label_file, bool useGPU, int channel, cv::Scalar mean, cv::Scalar stdev, const string& key)
	{
		m_size = size;
		m_iChannel = channel;
		m_bUseGPU = useGPU;
		m_mean = mean;
		m_stdev = stdev;
		std::stringstream sin(std::ios::binary | std::ios::out | std::ios::in);
		if (key != "")
		{
			ifstream fin(model_file, std::ios::binary | std::ios::in);
			char ch;
			int i = 0;
			while (fin.get(ch))
			{
				ch = ch ^ key[i >= key.size() ? i = 0 : i++];
				sin.put(ch);
			}
			fin.close();
		}
		m_pModel = key == "" ? std::make_shared<Model>(torch::jit::load(model_file)) : std::make_shared<Model>(torch::jit::load(sin));
		m_pModel->eval();
		m_pModel->to(torch::Device(useGPU ? torch::kCUDA : torch::kCPU), torch::kFloat32);
		ifstream classNamesFile(label_file);
		if (classNamesFile.is_open())
		{
			string className;
			while (std::getline(classNamesFile, className))
				m_vecClsNames.push_back(className);
			classNamesFile.close();
		}
	}

	TorchApp::ModelOutput TorchApp::forward(Mat img, bool swapRB)
	{
		try {
			if (m_stdev != cv::Scalar::all(1))
			{
				if (img.channels() != m_iChannel)
					throw exception(("expect input channel " + std::to_string(m_iChannel) + " but get " + std::to_string(img.channels())).c_str());
				img /= m_stdev;
				if (m_mean != cv::Scalar::all(0))
					m_mean /= m_stdev;
			}
			Mat blob = cv::dnn::blobFromImage(img, 1 / 255.0, m_size, m_mean * 255, swapRB);
			auto input_tensor = torch::from_blob(blob.data, { 1, m_iChannel, m_size.height, m_size.width });
			if (m_bUseGPU)
				input_tensor = input_tensor.to(at::kCUDA);
			auto result = m_pModel->forward({ input_tensor });
#ifndef TORCH_CPU
			if (m_bUseGPU)
				c10::cuda::getCurrentCUDAStream().synchronize();
#endif
			return result;
		}
		catch (const exception& e) {
			cerr << e.what() << endl;
			throw;
		}
	}

	TorchApp::ModelOutput TorchApp::forward(vector<Mat> imgs, bool swapRB)
	{
		try {
			Mat blob;
			if (m_stdev != cv::Scalar::all(1))
			{
				for (auto& img : imgs)
				{
					if (img.channels() != m_iChannel)
						throw exception(("expect input channel " + std::to_string(m_iChannel) + " but get " + std::to_string(img.channels())).c_str());
					img /= m_stdev;
					if (m_mean != cv::Scalar::all(0))
						m_mean /= m_stdev;
				}
			}
			blob = cv::dnn::blobFromImages(imgs, 1 / 255.0, m_size, m_mean * 255, swapRB);
			auto input_tensor = torch::from_blob(blob.data, { (int)imgs.size(), m_iChannel, m_size.height, m_size.width });
			if (m_bUseGPU)
				input_tensor = input_tensor.to(at::kCUDA);
			/*print(input_tensor[0].index({
				at::indexing::Slice(),
				at::indexing::Slice(25, 26),
				at::indexing::Slice() }));*/
			auto result = m_pModel->forward({ input_tensor });
#ifndef TORCH_CPU
			if (m_bUseGPU)
				c10::cuda::getCurrentCUDAStream().synchronize();
#endif
			return result;
		}
		catch (const exception& e) {
			cerr << e.what() << endl;
			throw;
		}
	}

}