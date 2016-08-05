// post-to-confidence.cc

// Copyright 2016  LeSpeech (author:  Xingyu Na)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Compute confidence score from posteriors for KWS\n"
        "following Guoguo's small KWS recipe. Smoothed post\n"
        "output is for debug. Difference is that label 0 is sil\n"
        "and label n-1 is filler.\n"
        "Usage: post-to-confidence [options] <post-rspecifier> "
        "<confidence-wspecifier> [<smooth-post-wspecifier>]\n";

    int32 w_smooth = 30;
    int32 w_max = 100;
    ParseOptions po(usage);
    po.Register("w-smooth", &w_smooth, "Posterior smoothing window length");
    po.Register("w-max", &w_max, "Confidence calculation window length");

    po.Read(argc, argv);

    if (po.NumArgs() != 2 && po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string post_rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);
    std::string post_wspecifier = po.GetArg(3);

    BaseFloatVectorWriter vec_writer(wspecifier);
    BaseFloatMatrixWriter post_writer(post_wspecifier);
    SequentialBaseFloatMatrixReader reader(post_rspecifier);
    int32 num_read = 0;
    for (; !reader.Done(); reader.Next(), num_read++) {
      std::string utt = reader.Key();
      const Matrix<BaseFloat> &post = reader.Value();

      const int32 frames = post.NumRows();
      const int32 labels = post.NumCols();
      Matrix<BaseFloat> p_smoothed(frames, labels);

      for (int32 i = 1; i < labels - 1; i++) {
        for (int32 j = 0; j < frames; j++) {
          int32 h_smooth = (0 > (j - w_smooth + 1)) ? 0 : (j - w_smooth + 1);
          BaseFloat sum = 0.0;
          for (int32 k = h_smooth; k <= j; k++)
            sum += post(k, i);
          KALDI_ASSERT(j - h_smooth + 1 > 0);
          p_smoothed(j, i) = sum / (j - h_smooth + 1);
        }
      }

      Vector<BaseFloat> confidence(frames);
      for (int32 j = 0; j < frames; j++) {
        int32 h_max = (0 > (j - w_max + 1)) ? 0 : (j - w_max + 1);
        BaseFloat mul = 1.0;
        for (int32 i = 1; i < labels - 1; i++) {
          BaseFloat maxp = -1000.0;
          for (int32 k = h_max; k <= j; k++)
            if (p_smoothed(k, i) > maxp)
              maxp = p_smoothed(k, i);
	  mul *= maxp;
        }
        confidence(j) = pow(mul, 1.0/(labels - 2));
      }
      vec_writer.Write(utt, confidence);
      if (post_writer.IsOpen())
        post_writer.Write(utt, p_smoothed);
    }
    return (num_read == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

