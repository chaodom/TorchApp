#ifndef TORCHAPP_TORCHCLASSIFY_H
#define TORCHAPP_TORCHCLASSIFY_H

#include "TorchApp.h"

namespace torchapp {

	typedef struct tagClassifyOutput
	{
		int cls_id;
		string cls_name;
		double score;
	}ClassifyOutput, * PClassifyOutput;


	class TORCHAPP_API TorchClassify :public TorchApp
	{
	public:	
		TorchClassify() {}
		virtual ~TorchClassify() {}

		vector<ClassifyOutput> classify(Mat img, int N = 1);
		vector<vector<ClassifyOutput>> classify(vector<Mat> imgs, int N = 1);

	};

}

#endif