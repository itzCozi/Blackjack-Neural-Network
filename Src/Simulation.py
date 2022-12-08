import numpy as np

def randomCard():
	card_arr = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
	return card_arr[np.random.randint(13)]

def cardValue(cardName):
	namesDict = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 10, 'Q': 10,
	             'K': 10}
	return namesDict[cardName]

def winIfHold(playerTotal, playerCards, dealerTotal):
	if playerCards >= 5: return 1  # five card charlie
	while True:
		if dealerTotal > 21:
			return 1  # player wins
		if dealerTotal > 16:
			if dealerTotal >= playerTotal:
				return 0  # Dealer Wins
			return 1
		else:
			dealerCard = randomCard()
			dealerTotal += cardValue(dealerCard)

def winIfDraw(playerTotal, playerCards, dealerTotal):
	playerCard = randomCard()
	playerTotal += cardValue(playerCard)
	playerCards += 1
	if playerTotal > 21:  # bust
		return 0
	if playerTotal > 16:
		if winIfHold(playerTotal, playerCards, dealerTotal): return 1
	else:
		if winIfDraw(playerTotal, playerCards, dealerTotal): return 1
	return 1

def runSimulation(playerCards_arr, dealerCard1, times, numCards):
	playerTotal = 0
	for card in playerCards_arr:
		playerTotal += cardValue(card)
	# playerTotal = cardValue(playerCard1) + cardValue(playerCard2)
	dealerCard2 = randomCard()
	dealerTotal = cardValue(dealerCard1) + cardValue(dealerCard2)
	holdWins = 0
	drawWins = 0
	for i in range(times):
		if winIfHold(playerTotal, numCards, dealerTotal):
			holdWins += 1
		if winIfDraw(playerTotal, numCards, dealerTotal):
			drawWins += 1

	if drawWins > holdWins:
		return [0, 1]  # left is hold , right is draw]
	else:
		return [1, 0]  # left is hold, right is draw
