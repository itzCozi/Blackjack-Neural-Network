import numpy as np

def randomCard():
	card_arr = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
	return card_arr[np.random.randint(13)]

def cardValue(cardName):
	namesDict = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 10, 'Q': 10,
	             'K': 10}
	return namesDict[cardName]

def winIfHold(playerTotal, playerCards, dealerTotal, dealer_cards):
	if playerCards >= 5: return True, dealer_cards, dealerTotal  # five card charlie
	while True:
		if dealerTotal > 21:
			return True, dealer_cards, dealerTotal  # player wins
		if dealerTotal > 16:
			if dealerTotal >= playerTotal:
				return False, dealer_cards, dealerTotal  # Dealer Wins
			return True, dealer_cards, dealerTotal
		else:
			dealerCard = randomCard()
			dealerTotal += cardValue(dealerCard)
			dealer_cards.append(dealerCard)

def Draw(playerTotal, playerCards, playerCards_arr, dealerTotal, dealer_cards):
	playerCard = randomCard()
	playerTotal += cardValue(playerCard)
	playerCards_arr.append(playerCard)
	playerCards += 1
	if playerTotal > 21:  # bust
		return 0
	if playerTotal > 16:
		if winIfHold(playerTotal, playerCards, dealerTotal): return 1
	else:
		if Draw(playerTotal, playerCards, dealerTotal): return 1
	return 1

def blackJackGames(neuralNet1, neuralNet2, NeuralNet2, times):
	numWins = 0
	numLoss = 0
	for time in range(times):
		playerCards_arr = [0, 0]
		playerCards_arr[0] = randomCard()
		playerCards_arr[1] = randomCard()
		dealerCards_arr = [0, 0]
		dealerCards_arr[0] = randomCard()
		dealerTotal = cardValue(dealerCards_arr[0])
		player_total = cardValue(playerCards_arr[0]) + cardValue(playerCards_arr[1])
		sample = [((player_total - 2) / 19.0), (dealerTotal - 1) / 9.0]
		guess = neuralNet1.feedforward(sample)
		action = None
		if float(guess[0]) < float(guess[1]):
			action = "Draw"
		else:
			action = "Hold"

		if action == "Hold":
			# dealer draws
			dealerCards_arr[1] = randomCard()
			dealerTotal += cardValue(dealerCards_arr[1])
			val, dealerCards_arr, dealerTotal = winIfHold(player_total, len(playerCards_arr), dealerTotal,
			                                              dealerCards_arr)
			if val:
				print("Game#", time, ": Player won!")
				print("Game#", time, ": Player Cards", playerCards_arr, "Total:", player_total)
				print("Game#", time, ": Dealer Cards", dealerCards_arr, "Total:", dealerTotal)
				numWins += 1
				continue
			else:
				print("Game#", time, ": Player lost :(")
				print("Game#", time, ": Player Cards", playerCards_arr, "Total:", player_total)
				print("Game#", time, ": Dealer Cards", dealerCards_arr, "Total:", dealerTotal)
				numLoss += 1
				continue
		if action == "Draw":
			playerCards_arr.append(randomCard())
			player_total += cardValue(playerCards_arr[2])
			if player_total == 21:
				print("Game#", time, ": Player Won :)")
				print("Game#", time, ": Player Cards", playerCards_arr, "Total:", player_total)
				print("Game#", time, ": Dealer Cards", dealerCards_arr, "Total:", dealerTotal)
				numWins += 1
				continue
			if player_total > 21:  # bust
				print("Game#", time, ": Player lost :(")
				print("Game#", time, ": Player Cards", playerCards_arr, "Total:", player_total)
				print("Game#", time, ": Dealer Cards", dealerCards_arr, "Total:", dealerTotal)
				numLoss += 1
				continue
			else:  # do you want to draw again or hold? ask the 2 neural network
				sample2 = [((player_total - 3) / 18.0), (dealerTotal - 1) / 9.0]
				guess = neuralNet2.feedforward(sample2)
				if float(guess[0]) < float(guess[1]):
					action = "Draw"
				else:
					action = "Hold"
				if action == "Hold":
					dealerCards_arr[1] = randomCard()
					dealerTotal += cardValue(dealerCards_arr[1])
					val, dealerCards_arr, dealerTotal = winIfHold(player_total, len(playerCards_arr), dealerTotal,
					                                              dealerCards_arr)
					if val:
						print("Game#", time, ": Player won!")
						print("Game#", time, ": Player Cards", playerCards_arr, "Total:", player_total)
						print("Game#", time, ": Dealer Cards", dealerCards_arr, "Total:", dealerTotal)
						numWins += 1
						continue
					else:
						print("Game#", time, ": Player lost :(")
						print("Game#", time, ": Player Cards", playerCards_arr, "Total:", player_total)
						print("Game#", time, ": Dealer Cards", dealerCards_arr, "Total:", dealerTotal)
						numLoss += 1
						continue
				if action == "Draw":
					playerCards_arr.append(randomCard())
					player_total += cardValue(playerCards_arr[3])
					if player_total == 21:
						print("Game#", time, ": Player Won :)")
						print("Game#", time, ": Player Cards", playerCards_arr, "Total:", player_total)
						print("Game#", time, ": Dealer Cards", dealerCards_arr, "Total:", dealerTotal)
						numWins += 1
						continue
					if player_total > 21:  # bust
						print("Game#", time, ": Player lost :(")
						print("Game#", time, ": Player Cards", playerCards_arr, "Total:", player_total)
						print("Game#", time, ": Dealer Cards", dealerCards_arr, "Total:", dealerTotal)
						numLoss += 1
						continue
					else:  # do you want to draw again or hold? ask the second neural network
						sample3 = [((player_total - 4) / 17.0), (dealerTotal - 1) / 9.0]
						guess = neuralNet2.feedforward(sample3)
						if float(guess[0]) < float(guess[1]):
							action = "Draw"
						else:
							action = "Hold"
						if action == "Hold":
							dealerCards_arr[1] = randomCard()
							dealerTotal += cardValue(dealerCards_arr[1])
							val, dealerCards_arr, dealerTotal = winIfHold(player_total, len(playerCards_arr),
							                                              dealerTotal, dealerCards_arr)
							if val:
								print("Game#", time, ": Player won!")
								print("Game#", time, ": Player Cards", playerCards_arr, "Total:", player_total)
								print("Game#", time, ": Dealer Cards", dealerCards_arr, "Total:", dealerTotal)
								numWins += 1
								continue
							else:
								print("Game#", time, ": Player lost :(")
								print("Game#", time, ": Player Cards", playerCards_arr, "Total:", player_total)
								print("Game#", time, ": Dealer Cards", dealerCards_arr, "Total:", dealerTotal)
								numLoss += 1
								continue
						if action == "Draw":  # now you have 5 cards so either you bust or you win
							playerCards_arr.append(randomCard())
							player_total += cardValue(playerCards_arr[4])
							if player_total > 21:  # bust
								print("Game#", time, ": Player lost :(")
								print("Game#", time, ": Player Cards", playerCards_arr, "Total:", player_total)
								print("Game#", time, ": Dealer Cards", dealerCards_arr, "Total:", dealerTotal)
								numLoss += 1
								continue

	print("Game Stats")
	print("Total games:", times)
	print("Total wins:", numWins)
	print("Total losses:", numLoss)
