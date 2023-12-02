# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
import random
import game
import util

class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal: return Directions.LEFT[left]
        return Directions.STOP

class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal: legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        scored = [(self.evaluationFunction(state), action) for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)
    
class AdvancedAgent(Agent):
    def __init__(self, evalFn="betterEvaluationFunction", depth=2):
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def getAction(self, state):
        # Minimax with alpha-beta pruning
        def minimax(state, depth, maximizing_player, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            legal_actions = state.getLegalPacmanActions()

            if maximizing_player:
                max_eval = float('-inf')
                for action in legal_actions:
                    successor_state = state.generateSuccessor(0, action)
                    eval = minimax(successor_state, depth - 1, False, alpha, beta)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break  # Beta cut-off
                return max_eval
            else:
                min_eval = float('inf')
                for action in legal_actions:
                    successor_state = state.generateSuccessor(0, action)
                    eval = minimax(successor_state, depth - 1, True, alpha, beta)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break  # Alpha cut-off
                return min_eval

        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        best_score = float('-inf')
        best_action = None

        for action in legal:
            successor_state = state.generateSuccessor(0, action)
            score = minimax(successor_state, self.depth - 1, False, float('-inf'), float('inf'))
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
          In my evaluation function I have divided the final score of the state in two parts
           1. When the ghosts are scared identified scaredTimes>0.
           2. Normal ghosts.
        Common evaluation score between both parts is the sum of the score for current score the steps
          for which the ghosts are scared, the reciprocal of the sum of food distance and number of foods eaten

          In the first case, from the sum I subtract the distance of the ghosts from current state
          and the number of power pellets, as the ghosts are currently in scared state. So closer pacman is to ghost better score

          In the second case since the ghosts are not scared hence distance to ghosts and number of power pellets
          are added to the sum.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    """ Manhattan distance to the foods from the current state """
    foodList = newFood.asList()
    from util import manhattanDistance
    foodDistance = [0]
    for pos in foodList:
        foodDistance.append(manhattanDistance(newPos,pos))

    """ Manhattan distance to each ghost from the current state"""
    ghostPos = []
    for ghost in newGhostStates:
        ghostPos.append(ghost.getPosition())
    ghostDistance = [0]
    for pos in ghostPos:
        ghostDistance.append(manhattanDistance(newPos,pos))

    numberofPowerPellets = len(currentGameState.getCapsules())

    score = 0
    numberOfNoFoods = len(newFood.asList(False))           
    sumScaredTimes = sum(newScaredTimes)
    sumGhostDistance = sum (ghostDistance)
    reciprocalfoodDistance = 0
    if sum(foodDistance) > 0:
        reciprocalfoodDistance = 1.0 / sum(foodDistance)
        
    score += currentGameState.getScore()  + reciprocalfoodDistance + numberOfNoFoods

    if sumScaredTimes > 0:    
        score +=   sumScaredTimes + (-1 * numberofPowerPellets) + (-1 * sumGhostDistance)
    else :
        score +=  sumGhostDistance + numberofPowerPellets
    return score


# Abbreviation
better = betterEvaluationFunction


def scoreEvaluation(state):
    return state.getScore()
