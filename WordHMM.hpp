/*
 * WordHMM.hpp
 *
 *  Created on: Jan 13, 2014
 *      Author: yuncong
 */

#ifndef WORDHMM_HPP_
#define WORDHMM_HPP_

#include <armadillo>
#include <mlpack/core.hpp>
//#include <mlpack/core/dists/discrete_distribution.hpp>
#include <mlpack/core/dists/gaussian_distribution.hpp>
#include <mlpack/methods/hmm/hmm.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>

#include <queue>

using namespace std;
using namespace boost;
using namespace arma;
using namespace mlpack;
using namespace mlpack::hmm;
using namespace mlpack::distribution;

//typedef HMM<GaussianDistribution> GHMM;

namespace mlpack {
namespace hmm {

//template<typename Distribution>
//class HMMWithMemory: public HMM<Distribution> {
//
//};

template<typename Distribution>
class HMMExtended: public HMM<Distribution> {
public:
	HMMExtended(const arma::mat& transition,
			const std::vector<Distribution>& emission, int stage_per_letter,
			int penup_n, int n) :
			HMM<Distribution>(transition, emission), transition(transition), emission(
					emission), dimensionality(emission[0].Dimensionality()), delta(
					1000), stage_per_letter(stage_per_letter), penup_n(penup_n), n(
					n) {
		v = zeros<vec>(transition.n_cols);
	}

	void AddObservation(const vec& p) {
		observations.insert_cols(observations.n_cols, p);

		ivec best_predecessor = -1 * ones<ivec>(transition.n_cols);
		vec v_new = -datum::inf * ones<vec>(transition.n_cols);

		vec diff, exponent;
		mat cov;
		double emit_prob;

		if (observations.n_cols == 1) {
			// first point
			for (size_t j = 0; j < transition.n_cols; j++) {
				if (j % 8 == 0 && j < stage_per_letter * n) {

					diff = emission[j].Mean() - p;
					diff = diff(0);
					cov = emission[j].Covariance()(0, 0);

					exponent = -0.5 * (trans(diff) * inv(cov) * diff);
					emit_prob = pow(2 * M_PI, (double) (diff.n_elem) / -2.0)
							* pow(det(cov), -0.5) * exp(exponent[0]);

					v[j] = log(1.0 / n * emit_prob);
					beam.push_back(j);
				}
			}
		} else {
			BOOST_FOREACH(uint i, beam) {
				//			cout << "beam node " << i << endl;
				for (size_t j = 0; j < transition.n_cols; j++) {
					if (transition(i, j) > 0) {
						//				if (transition(i, j) > 0 && emission[j].Probability(p) > 0.01) {
						//					cout  << i << " , " << j << endl;
						if (j >= stage_per_letter * n) {
							// penup states
							diff = emission[j].Mean() - p;
							diff = diff.subvec(1, 2);
							cov = emission[j].Covariance().submat(1, 1, 2, 2);
						} else if (j % 8 == 1) {
							// initial stage
							diff = emission[j].Mean() - p;
							diff = diff(0);
							cov = emission[j].Covariance()(0, 0);
						} else {
							diff = emission[j].Mean() - p;
							cov = emission[j].Covariance();
						}
						exponent = -0.5 * (trans(diff) * inv(cov) * diff);
						emit_prob = pow(2 * M_PI, (double) (diff.n_elem) / -2.0)
								* pow(det(cov), -0.5) * exp(exponent[0]);

						double vij = v(i) + log(transition(i, j) * emit_prob);
						//					printf("%f, %f, %f\n", v(i), transition(i, j), emission[j].Probability(p)) ;
						if (vij > v_new(j)) {
							v_new(j) = vij;
							best_predecessor(j) = i;
							//					cout << "vij " << vij << endl;
						}
					}
				}
			}
			//		best_predecessor.print();
			//		v_new.print();

			double v_best = v_new.max();
			std::vector<uint> beam_new;
			for (size_t j = 0; j < transition.n_cols; j++) {
				if (v_new(j) >= v_best - delta) {
					beam_new.push_back(j);
				}
			}
			v = v_new;
			beam = beam_new;
		}

		printf("beam size = %d\n", beam.size());

//		cout << backtrack.n_cols << endl;

		backtrack.insert_cols(backtrack.n_cols, best_predecessor);
//		if (backtrack.n_cols > 2) {
//			backtrack.col(1).print();
//		}
	}

	double backtracking(uvec & sequence) {
		sequence.set_size(observations.n_cols);
		uword index;
		double best_likelihood = v.max(index);
//		cout << index << endl;
		sequence[backtrack.n_cols - 1] = index;
		for (int t = backtrack.n_cols - 2; t >= 0; t--) {
//			cout << "t=" << t << endl;
//			seq.print();
//			cout << "bt[t]=" << backtrack.unsafe_col(t)(seq[t + 1]) << endl;
//			cout << seq[t + 1] << endl;
//			cout << backtrack.col(t);
//			ivec r = backtrack.col(t);
//			seq[t] = r(seq[t + 1]);
			sequence[t] = backtrack.unsafe_col(t + 1)(sequence[t + 1]);
		}

		return best_likelihood;
	}

//	// something wrong with the way the optimal state sequence is computed; there is no backtracking.
//	double PredictBeamSearch(const arma::mat& dataSeq,
//			arma::Col<size_t>& stateSeq) const {
//		// This is an implementation of the Viterbi algorithm for finding the most
//		// probable sequence of states to produce the observed data sequence.
//		// This extends the original mlpack implementation with beam search.
//		stateSeq.set_size(dataSeq.n_cols);
//		arma::mat logStateProb(transition.n_rows, dataSeq.n_cols);
//
//		// Store the logs of the transposed transition matrix.  This is because we
//		// will be using the rows of the transition matrix.
//		arma::mat logTrans(log(trans(transition)));
//
//		// The calculation of the first state is slightly different; the probability
//		// of the first state being state j is the maximum probability that the state
//		// came to be j from another state.
//		logStateProb.col(0).zeros();
//		for (size_t state = 0; state < transition.n_rows; state++)
//			logStateProb[state] = log(
//					transition(state, 0)
//							* emission[state].Probability(
//									dataSeq.unsafe_col(0)));
//
//		// Store the best first state.
//		arma::uword index;
//		logStateProb.unsafe_col(0).max(index);
//		stateSeq[0] = index;
//
//		for (size_t t = 1; t < dataSeq.n_cols; t++) {
//			// Assemble the state probability for this element.
//			// Given that we are in state j, we state with the highest probability of
//			// being the previous state.
//			for (size_t j = 0; j < transition.n_rows; j++) {
//				arma::vec prob = logStateProb.col(t - 1) + logTrans.col(j);
//				logStateProb(j, t) = prob.max()
//						+ log(emission[j].Probability(dataSeq.unsafe_col(t)));
//			}
//
//			// Store the best state.
//			logStateProb.unsafe_col(t).max(index);
//			stateSeq[t] = index;
//		}
//
//		return logStateProb(stateSeq(dataSeq.n_cols - 1), dataSeq.n_cols - 1);
//	}

private:
	arma::mat transition;
	std::vector<Distribution> emission;
	uint dimensionality;
	std::vector<uint> beam;
	double delta;
	vec v;
	imat backtrack;
	arma::mat observations;
	int stage_per_letter;
	int penup_n;
	int n;
}
;
}
}

typedef HMMExtended<GaussianDistribution> GHMM;

#endif /* WORDHMM_HPP_ */
