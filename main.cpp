/*
 * main.cpp
 *
 *  Created on: Jan 9, 2014
 *      Author: yuncong
 */

#include <stdlib.h>     /* srand, rand */
#include <iostream>
#include <fstream>

#include <armadillo>
#include <mlpack/core.hpp>
//#include <mlpack/core/dists/discrete_distribution.hpp>
#include <mlpack/core/dists/gaussian_distribution.hpp>
#include <mlpack/methods/hmm/hmm.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/range/irange.hpp>
#include <boost/timer.hpp>
#include <boost/progress.hpp>

#include "WordHMM.hpp"

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace arma;
using namespace mlpack;
using namespace mlpack::hmm;
using namespace mlpack::distribution;

inline int ord(char c) {
	return int(c) - int('a');
}

inline char cha(int l) {
	return char(l + int('a'));
}

//string interpret_sequence(uvec sequence, int stage_per_letter) {
//	int curr_stage = -1, prev_stage;
//	char curr_letter;
//	string prediction = "";
//	for (size_t i = 0; i < sequence.n_elem; i++) {
//		curr_letter = cha(sequence[i] / stage_per_letter);
//		prev_stage = curr_stage;
//		curr_stage = sequence[i] % stage_per_letter;
//		cout << curr_letter << curr_stage;
//		if (prev_stage != 0 && curr_stage == 0) {
//			prediction += curr_letter;
//		}
//	}
//	cout << endl;
//	return prediction;
//}

string interpret_sequence(uvec sequence, int stage_per_letter,
		string& readable_seq) {
	int curr_stage = -1, prev_stage;
	char curr_letter;
	stringstream readable_seq_ss;
	string prediction = "";
	for (size_t i = 0; i < sequence.n_elem; i++) {
		curr_letter =
				sequence[i] >= stage_per_letter * 26 ?
						'-' : cha(sequence[i] / stage_per_letter);
		prev_stage = curr_stage;
		curr_stage = sequence[i] % stage_per_letter;
		readable_seq_ss << curr_letter << curr_stage;
		if (prev_stage != 0 && curr_stage == 0 && curr_letter != '-') {
			prediction += curr_letter;
		}
	}
	readable_seq = readable_seq_ss.str();
	return prediction;
}

int main() {
	bool not_use_x = true;

	int n = 26;
	int stage_per_letter = 8;
	int penup_n = 8;
	int state_n = stage_per_letter * n + penup_n * n;
	mat mean, cov;
	string filename;
	vector<GaussianDistribution> emission;
	GaussianDistribution dist;

	for (size_t i = 0; i < n; i++) {
		filename = str(format("hmm_params/mean_%d.txt") % cha(i));
		mean.load(filename, raw_ascii);
		if (not_use_x) {
			mean = mean.cols(1, 3);
		}
		mean = mean.t();
		filename = str(format("hmm_params/cov_%d.txt") % cha(i));
		cov.load(filename, raw_ascii);
		if (not_use_x) {
			cov = cov.cols(1, 3);
		}
		cov = cov.t();

		for (size_t j = 0; j < stage_per_letter; j++) {
			dist.Mean() = mean.col(j);
			dist.Covariance() = diagmat(cov.col(j));
			emission.push_back(dist);
		}
	}

	for (size_t letter = 0; letter < n; letter++) {
		for (size_t penup_direction = 0; penup_direction < penup_n;
				penup_direction++) {
			rowvec mean, cov;
			if (penup_direction == 0) {
				mean << NAN << 1 << 0 << endr;
			} else if (penup_direction == 1) {
				mean << NAN << 1 / sqrt(2) << 1 / sqrt(2) << endr;
			} else if (penup_direction == 2) {
				mean << NAN << 0 << 1 << endr;
			} else if (penup_direction == 3) {
				mean << NAN << -1 / sqrt(2) << 1 / sqrt(2) << endr;
			} else if (penup_direction == 4) {
				mean << NAN << -1 << 0 << endr;
			} else if (penup_direction == 5) {
				mean << NAN << -1 / sqrt(2) << -1 / sqrt(2) << endr;
			} else if (penup_direction == 6) {
				mean << NAN << 0 << -1 << endr;
			} else if (penup_direction == 7) {
				mean << NAN << 1 / sqrt(2) << -1 / sqrt(2) << endr;
			}
			cov << NAN << 0.1 << 0.1 << endr;
			dist.Mean() = mean.t();
			dist.Covariance() = diagmat(cov.t());
			emission.push_back(dist);
		}
	}

	mat bigram;
	bigram.load("bigram", raw_ascii);

	bool single_letter = false;

//	double next_prob = 0.01;
//	double self_prob = 0.99;
	double next_prob = 0.2;
	double self_prob = 0.8;

	mat transition(state_n, state_n);
//	for (size_t curr_letter = 0; curr_letter < n; curr_letter++) {
//		for (size_t stage = 0; stage < stage_per_letter; stage++) {
//			int state = curr_letter * stage_per_letter + stage;
//			if (stage == stage_per_letter - 1) { //final stage
//			// bigram transitions
//				for (size_t next_letter = 0; next_letter < n; next_letter++) {
//					if (single_letter) {
//						transition(state, next_letter * stage_per_letter) = 0;
//						transition(state, state) = 1;
//					} else {
//						transition(state, next_letter * stage_per_letter) =
//								next_prob * bigram(curr_letter, next_letter);
//						transition(state, state) = self_prob;
//					}
//				}
//			} else {
//				transition(state, state + 1) = next_prob;
//				transition(state, state) = self_prob;
//			}
//		}
//	}

	for (size_t curr_letter = 0; curr_letter < n; curr_letter++) {
		for (size_t stage = 0; stage < stage_per_letter; stage++) {
			int state = curr_letter * stage_per_letter + stage;
			if (stage == stage_per_letter - 1) {
				// final stage
				for (int penup_direction = 0; penup_direction < penup_n;
						penup_direction++) {
					int penup_state = stage_per_letter * n
							+ curr_letter * penup_n + penup_direction;
					transition(state, penup_state) = next_prob / penup_n;
					transition(penup_state, penup_state) = self_prob;
					for (size_t next_letter = 0; next_letter < n;
							next_letter++) {
						int next_letter_begin = next_letter * stage_per_letter;
						transition(penup_state, next_letter_begin) = next_prob
								* bigram(curr_letter, next_letter);
					}
				}
			} else {
				// internal stage
				transition(state, state + 1) = next_prob;
				transition(state, state) = self_prob;
			}
		}
	}

//	transition.print();

// load letter prototypes
//	vector<vector<mat> > inks;
//	mat ink;
//	for (size_t i = 0; i < n; i++) {
//		path dir_path(str(format("user_data/%d") % i));
//		directory_iterator end_itr;
//		vector<mat> inks_single_letter;
//		for (directory_iterator itr(dir_path); itr != end_itr; ++itr) {
//			path p(itr->path());
//			ink.load(p.string());
//			ink = ink.t();
//			inks_single_letter.push_back(ink);
//		}
//		inks.push_back(inks_single_letter);
//	}

	vector<string> common_words;

// 1. manually specify test words
	common_words.push_back("thief");

// 2. automatically load test words from common_words
//	std::ifstream infile;
//	infile.open("common_words");
//	if (infile) {
//		string s = "";
//		while (getline(infile, s)) {
//			if (s.size() == 5)
//				common_words.push_back(s);
//		}
//	}

	double avg_accuracy = 0;
	vector<double> accuracy_list;

	srand(time(NULL));

	BOOST_FOREACH(string truth, common_words) {

		int correct = 0;

		int NUM_TRIAL = 1;
		for (int trial = 0; trial < NUM_TRIAL; trial++) {

			create_directories(
					path(str(format("predictions/%s") % truth.c_str())));
			filename = str(
					format("predictions/%s/%d.txt") % truth.c_str() % trial);
			FILE * outputFile = fopen(filename.c_str(), "w");

			mat word_ink;

			// 1. construct word ink from random prototypes
//			for (int i = 0; i < truth.length(); i++) {
//				vector<mat> single_letter = inks[ord(truth[i])];
//				mat chosen_prototype = single_letter[rand() % single_letter.size()];
//				if (i == 0) {
//					word_ink = chosen_prototype;
//				} else {
//					word_ink = join_rows(word_ink, chosen_prototype);
//				}
//			}

// or 2. load word ink directly generated from python script
			word_ink.load(
					str(
							format("random_words/%s/%d.txt") % truth.c_str()
									% trial));
			if (not_use_x) {
				word_ink = word_ink.cols(1,3);
			}
			word_ink = word_ink.t();

			GHMM model = GHMM(transition, emission, stage_per_letter, penup_n,
					n);

			string prediction;
			double likelihood;
			uvec sequence;
			string readable_seq;

			for (size_t t = 0; t < word_ink.n_cols; t++) {
				{
//					cout << "t = " << t << endl;
//					progress_timer timer; // start timing
					model.AddObservation(word_ink.col(t));

					likelihood = model.backtracking(sequence);
					prediction = interpret_sequence(sequence, stage_per_letter,
							readable_seq);
					printf("%s\n", readable_seq.c_str());
					printf("Trial %d, prediction = %s, likelihood = %f\n",
							trial, prediction.c_str(), likelihood);
				}
			}

//			sequence.print();
			if (prediction == truth) {
				correct++;
			}
			fprintf(outputFile, "%s\n", readable_seq.c_str());
			fclose(outputFile);
		}

		double accuracy = (float) correct / NUM_TRIAL;
		accuracy_list.push_back(accuracy);
		avg_accuracy += accuracy;

		printf("Over %d randomizations, correct rate for %s = %f\n", NUM_TRIAL,
				truth.c_str(), accuracy);
	}

	vec ac = vec(accuracy_list);
//	uvec hc = hist(ac, 100);
//	hc.print();
	ac.save("accuracy_list");

	printf("Average accuracy for %d common words = %f\n", common_words.size(),
			avg_accuracy / common_words.size());
//	fprintf(outputFile, "Average accuracy for %d common English words = %f\n",
//			common_words.size(), avg_accuracy / common_words.size());
}
