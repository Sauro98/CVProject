#ifndef __ANN_HPP_INCLUDED__
#define __ANN_HPP_INCLUDED__

#include <vector>
#include <string>
#include <mutex>

/** \brief Class used for training and using an ANN (Artificial Neural Network). */
class Ann
{
public:

/** \brief Class constructor which lets loading the ANN from a .ann file. */
/** The ".ann" extension is added internally. */
/** \param name fileName of the file from which the ANN will be loaded.
  * \see Save(std::string fileName) */
	Ann(std::string fileName);

/** \brief Class constructor which lets creating a new ANN from the number of neuron per layer. */
/** \param NPL Vector which specifies the number of neurons and layers the new ANN should have.\n */
/** For instance, this code would generate an ANN with three layers of respectively two, four and one neurons:
    ~~~~~~~~~~~~~{.cpp}
    std::vector<unsigned int> myVector;
    myVector.push_back(2);
    myVector.push_back(4);
    myVector.push_back(1);
    Ann myAnn(myVector);
    ~~~~~~~~~~~~~ */
	Ann(const std::vector<unsigned int> &NPL);

/** \brief Randomizes the ANN, setting new random weights. Called e.g. by Ann(const std::vector<unsigned int> &NPL). */
/** \param seed Seed for the random number generator. */
	void randomize(unsigned int seed);

/** \brief Saves the ANN on a .ann file. */
/** Lets you saving the ANN in a text format. */
/** \param fileName Name of the file. The ".ann" extension is added internally. Default: "NewAnn".
  * \return True on success, false on failure.
  * \see  Ann(std::string fileName);*/
	bool save(std::string fileName = "NewAnn");

/** \brief Shows the ANN. */
/** Used to show all the ANN data. Mostly for debugging purposes. */
	void show();

/** \brief Trains the ANN. */
/** Lets you train the ANN, using somewhat standard online SGD with momentum. */
/** \param inputs training inputs.
  * \param outputs training targets.
  * \param vInputs validation inputs.
  * \param vOutputs validation targets.
  * \param minError minimum error to reach before automatic stopping.
  * \param iterations max number of iterations.
  * \param minErrorEpochs number of epochs between subsequent validation error test.
  * \param patience patience for early stopping. -1 disables early stopping.
  * \param filename base filename for saving at each iteration. Default: don't save.
  * \param useCategorical use categorical error instead of MSE.
  * \param verbose verbosity.
  * \param N Learning rate.
  * \param B Momentum factor.
  * \param mtx Optional mutex for "cout"s.
  * \param threadIndex Optional index for "cout"s.
  * \return number of iterations performed. */
	unsigned int train(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs,
					   const std::vector<std::vector<double>> &vInputs, const std::vector<std::vector<double>> &vOutputs,
					   double minError = 0.01, unsigned int iterations = 1000, unsigned int minErrorEpochs = 10,
					   int patience = -1, std::string filename = "",
					   bool useCategorical = false, bool verbose = true,
					   double N = 0.3, double B = 0.1, std::mutex* mtx = nullptr, unsigned int threadIndex = 0);

/** \brief Get error. */
/** Computes the error of the ANN on inputs w.r.t. outputs. */
/** \param inputs test inputs.
  * \param outputs test targets.
  * \return You guessed it, the error. */
	double getError(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs);

/** \brief Get categorical error. */
/** Computes the ratio in [0-1] of inputs wrongly classified by the ANN, assuming one-hot encoding.
  * The lower the better. */
/** \param inputs test inputs.
  * \param outputs test targets.
  * \return The ratio of wrongly classified inputs. */
	double getCategoricalError(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs);

/** \brief Use the ANN. */
/** Computes the output of the ANN on the given input. */
/** \param inputs the inputs.
  * \param outputs the output of the ANN. */
	void use(const std::vector<double> &inputs, std::vector<double> &outputs);

/** \brief Use the ANN. */
/** Computes the output of the ANN on the given input, converting the result from one-hot to an index. */
/** \param inputs the inputs.
  * \return the category index. */
	unsigned int useCategorical(const std::vector<double> &inputs);

private:
	std::vector<unsigned int> NpL = std::vector<unsigned int>();
	std::vector<std::vector<std::vector<double>>> WpL = std::vector<std::vector<std::vector<double>>>();
	std::vector<std::vector<double>> results = std::vector<std::vector<double>>();
};

#endif
