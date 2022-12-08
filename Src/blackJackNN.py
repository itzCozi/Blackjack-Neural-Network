import time
import numpy as np
import os
from Src import Gameplay as game
from Src import Simulation as sim

prt_debug = False
CC = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')

def randomCard():
	card_arr = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
	return card_arr[np.random.randint(13)]


def cardValue(cardName):
	namesDict = {
		'A': 1,
		'2': 2,
		'3': 3,
		'4': 4,
		'5': 5,
		'6': 6,
		'7': 7,
		'8': 8,
		'9': 9,
		'10': 10,
		'J': 10,
		'Q': 10,
		'K': 10
	}
	return namesDict[cardName]


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


class NeuralNetwork:
	input_nodes = 0
	hidden_nodes = 0
	output_nodes = 0
	weights_ih = []
	weights_ho = []
	bias_h = 0
	bias_o = 0
	learning_rate = 0.01

	def __init__(self, numI, numH, numO):
		self.input_nodes = numI
		self.hidden_nodes = numH
		self.output_nodes = numO
		self.weights_ih = np.random.uniform(-1, 1, size=(numH, numI))  # 🦧
		self.weights_ho = np.random.uniform(-1, 1, size=(numO, numH))  # 🦍
		self.bias_h = np.zeros((numH, 1))
		self.bias_h.fill(np.random.uniform(-1, 1))
		self.bias_o = np.zeros((numO, 1))
		self.bias_o.fill(np.random.uniform(-1, 1))
		print('Neural Net was initialized')
		print('#input nodes:', self.input_nodes, '#hidden nodes:',
		      self.hidden_nodes, '#outputs:', self.output_nodes, '\n')

	def feedforward(self, inputs):
		if prt_debug:
			print('FEED FORWARD STARTING\n')
		inputs = np.array(inputs).reshape(self.input_nodes, 1)
		if prt_debug:
			print('Inputs:\n', inputs, '\n')
		hidden = self.weights_ih.dot(inputs)
		if prt_debug:
			print('Hiddens (weights_ih * inputs) :\n', hidden, '\n')
		hidden = hidden + self.bias_h
		if prt_debug:
			print('Hiddens (hiddens + self.bias_h) :\n', hidden, '\n')
		hidden = sigmoid(hidden)
		if prt_debug:
			print('Hiddens (sidmoid(hidden)) :\n', hidden, '\n')
		output = self.weights_ho.dot(hidden)
		if prt_debug:
			print('Outputs (weights_ho.dot(hidden)) :\n', output, '\n')
		output = output + self.bias_o
		if prt_debug:
			print('Outputs (output + self.bias_o) :\n', output, '\n')
		# activation function
		output = sigmoid(output)
		if prt_debug:
			print('Outputs (sigmoid(output)) :\n', output, '\n')

		return output

	def trainBP(self, inputs, targets):
		targets = np.array(targets).reshape(self.output_nodes, 1)
		if prt_debug:
			print('Targets:\n', targets, '\n')
		inputs = np.array(inputs).reshape(self.input_nodes, 1)
		if prt_debug:
			print('Inputs:\n', inputs, '\n')
		outputs = self.feedforward(inputs)
		if prt_debug:
			print('!!!Outputs:(self.feedforward(inputs)\n', outputs, '\n')

		# do the output error ❌
		output_errors = targets - outputs
		if prt_debug:
			print('!!!Output Errors:(targets - outputs)\n', output_errors, '\n')

		# do the um the hidden layer errors
		hidden_errors = self.weights_ho.T.dot(output_errors)
		if prt_debug:
			print(
				'Hidden BXgs2AAWD7gF2WUNhva7byzkpR4QbvM3YHoCJzrGebnb hidden_errors' +
				"\n")

		gradients = outputs * (1 - outputs)
		gradients = gradients * (output_errors)
		gradients = gradients.dot(self.learning_rate)  # sigmoid cost function
		if prt_debug:
			print('Gradients:(derivative *learning_rate)\n', gradients, '\n')

		weight_ho_deltas = gradients.dot(hidden_errors.T)
		if prt_debug:
			print('weight_ho_deltas:(gradients.dot(hidden_errors.T))\n',
			      weight_ho_deltas, '\n')

		# adjust weights by deltas
		self.weights_ho += weight_ho_deltas
		if prt_debug:
			print('weights_ho:(weights_ho += weight_ho_deltas)\n', self.weights_ho,
			      '\n')
		# adjust bias by deltas
		self.bias_o += gradients
		if prt_debug:
			print('bias_o:(bias_o += gradients )\n', self.bias_o, '\n')

		# code copied from feedforward to access hidden variable 😳
		hidden = self.weights_ih.dot(inputs)
		hidden = hidden + self.bias_h
		hidden = sigmoid(hidden)

		# calculate hidden gradient
		hidden_gradients = hidden * (1 - hidden)
		hidden_gradients = hidden_gradients * hidden_errors
		hidden_gradients = hidden_gradients.dot(self.learning_rate)
		if prt_debug:
			print('hidden_gradients :((derivative *learning_rate) )\n',
			      hidden_gradients, '\n')
		weight_ih_deltas = hidden_gradients.dot(inputs.T)
		if prt_debug:
			print('weight_ih_deltas :((hidden_gradients.dot(inputs.T)) )\n',
			      weight_ih_deltas, '\n')
		self.weights_ih += weight_ih_deltas
		self.bias_h += hidden_gradients


# Second network training with 3 cards
def trainNN3Cards(nn2, epochs):
	print("Training Neural Network 2")

	desiredOutput = None

	# playerCards
	playerCard = [0, 0, 0]

	for epoch in range(epochs):
		playerCard[0] = randomCard()
		playerCard[1] = randomCard()
		playerCard[2] = randomCard()
		dealerCard = randomCard()
		dealerTotal = cardValue(dealerCard)
		playerTotal = cardValue(playerCard[0]) + cardValue(
			playerCard[1]) + cardValue(playerCard[2])
		sample = [((playerTotal - 3) / 18.0), (dealerTotal - 1) / 9.0]
		desiredOutput = sim.runSimulation(playerCard, dealerCard, 20, 3)
		# Perdict statment
		nn2.trainBP(sample, desiredOutput)
	return nn2


# Third network training with 4 cards yuhhh 🤙
def trainNN4Cards(nn3, epochs):
	CC()
	print("Training Neural Network 3")
	desiredOutput = None

	# playerCards
	playerCard = [0, 0, 0, 0]

	for epoch in range(epochs):
		playerCard[0] = randomCard()
		playerCard[1] = randomCard()
		playerCard[2] = randomCard()
		playerCard[3] = randomCard()
		dealerCard = randomCard()
		dealerTotal = cardValue(dealerCard)
		playerTotal = cardValue(playerCard[0]) + cardValue(
			playerCard[1]) + cardValue(playerCard[2]) + cardValue(playerCard[3])
		sample = [((playerTotal - 4) / 17.0), (dealerTotal - 1) / 9.0]
		desiredOutput = sim.runSimulation(playerCard, dealerCard, 20, 3)
		# Perdict statment
		nn3.trainBP(sample, desiredOutput)
	return nn3


def doit(epochs, showFrequency):
	guess = None
	confidence = None
	epoch = 1
	desiredOutput = None
	nn = NeuralNetwork(2, 6, 2)
	nn2 = NeuralNetwork(2, 6, 2)
	nn2 = trainNN3Cards(nn2, 100000)
	nn3 = NeuralNetwork(2, 6, 2)
	nn3 = trainNN4Cards(nn3, 100000)

	# values for NN1
	totalRight = 0
	totalWrong = 0
	# values for NN2
	totalRight2 = 0
	totalWrong2 = 0
	# values for NN3
	totalRight3 = 0
	totalWrong3 = 0

	# playerCards
	playerCard = [0, 0]

	for epoch in range(epochs):

		playerCard[0] = randomCard()
		playerCard[1] = randomCard()
		playerTotal = cardValue(playerCard[0]) + cardValue(playerCard[1])
		dealerCard = randomCard()
		dealerTotal = cardValue(dealerCard)

		sample = [((playerTotal - 2) / 19.0), (dealerTotal - 1) / 9.0]
		desiredOutput = sim.runSimulation(playerCard, dealerCard, 20, 2)
		guess = nn.feedforward(sample)

		desired = None
		if desiredOutput[0] == 1:
			desired = "Hold"
		else:
			desired = "Draw"
		guessAction = None
		confidence = None

		if float(guess[0]) < float(guess[1]):
			guessAction = "Draw"
			confidence = float(guess[1]) - float(guess[0])

		else:
			confidence = float(guess[0]) - float(guess[1])
			guessAction = "Hold"

		if guessAction == desired:
			totalRight += 1
		else:
			totalWrong += 1

		if epoch <= 5 or epoch % showFrequency == 0:
			conf = "{:0.2f}".format(confidence)
			# Long asf print statment
			print(
				epoch,
				':'
				'cards: P:',
				playerCard,
				'D:',
				'[',
				dealerCard,
				']',
				"NN Move:",
				guessAction,
			)

		nn.trainBP(sample, desiredOutput)  # adjust weights

		# if draw then draw and do training the second neural network
		if guessAction == "Draw":
			if epoch <= 5 or epoch % showFrequency == 0:
				print("Draw a 3th Card")
			playerCard.append(randomCard())
			playerTotal += cardValue(playerCard[2])
			sample2 = [((playerTotal - 3) / 18.0), (dealerTotal - 1) / 9.0]
			desiredOutput2 = sim.runSimulation(playerCard, dealerCard, 20, 3)
			guess2 = nn2.feedforward(sample2)

			desired = None
			if desiredOutput2[0] == 1:
				desired = "Hold"
			else:
				desired = "Draw"
			guessAction = None
			confidence = None
			if float(guess2[0]) < float(guess2[1]):
				guessAction = "Draw"
				confidence = float(guess2[1]) - float(guess2[0])

			else:
				confidence = float(guess2[0]) - float(guess2[1])
				guessAction = "Hold"

			if guessAction == desired:
				totalRight2 += 1
			else:
				totalWrong2 += 1

			if epoch <= 5 or epoch % showFrequency == 0:
				conf = "{:0.2f}".format(confidence)
				# Long Ahh print statment
				print(epoch, ':(NN2)'
				             'cards: P:', playerCard, 'D:', '[', dealerCard, ']', "NN Move:",
				      guessAction)
			if guessAction == "Draw":
				if epoch <= 5 or epoch % showFrequency == 0:
					print("Draw a 4th Card")
				playerCard.append(randomCard())
				playerTotal += cardValue(playerCard[3])
				sample2 = [((playerTotal - 4) / 17.0), (dealerTotal - 1) / 9.0]
				desiredOutput2 = sim.runSimulation(playerCard, dealerCard, 20, 3)
				guess2 = nn3.feedforward(sample2)

				desired = None
				if desiredOutput2[0] == 1:
					desired = "Hold"
				else:
					desired = "Draw"
				guessAction = None
				confidence = None
				if float(guess2[0]) < float(guess2[1]):
					guessAction = "Draw"
					confidence = float(guess2[1]) - float(guess2[0])

				else:
					confidence = float(guess2[0]) - float(guess2[1])
					guessAction = "Hold"

				if guessAction == desired:
					totalRight3 += 1
				else:
					totalWrong3 += 1
				if epoch <= 5 or epoch % showFrequency == 0:
					# Conf variable
					conf = "{:0.2f}".format(confidence)
					# Long Ahh print statment
					print(epoch, ':(NN3)'
					             'cards: P:', playerCard, 'D:', '[', dealerCard, ']',
					      "NN Move:", guessAction)
				playerCard.pop(3)
			playerCard.pop(2)
	print("Neural Networks have been trained, these are their accuracy rates:")
	print("NN1:", (100.0 * totalRight) / (totalRight + totalWrong))
	print("NN2:", (100.0 * totalRight2) / (totalRight2 + totalWrong2))
	print("NN3:", (100.0 * totalRight3) / (totalRight3 + totalWrong3))
	CC()
	print('------Game Play------')
	howmanygames = input('How many games would you like to play? ')
	time.sleep(2)
	# Call to game to display it
	game.blackJackGames(nn, nn2, nn2, int(howmanygames))


doit(100000, 10000)
