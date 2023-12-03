# ghostAgents.py
# --------------
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


from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util


class GhostAgent(Agent):
    def __init__(self, index):
        super().__init__(index)
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    "A ghost that chooses a legal action uniformly at random."

    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index): dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(self, index, prob_attack=0.8, prob_scaredFlee=0.8):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):
        # Read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += (1 - bestProb) / len(legalActions)
        dist.normalize()
        return dist


class AlphaBetaGhost(GhostAgent):
    def __init__(self, index, depth='3'):
        super().__init__(index)
        self.index = index
        self.depth = int(depth)

    def getDistribution(self, state):
        action = self.minimax(state, self.index, self.depth)[1]
        dist = util.Counter()
        dist[action] = 1.0
        return dist

    def minValue(self, state, agentIndex, depth, alpha, beta):
        actions = []
        for action in state.getLegalActions(agentIndex):
            v = self.minimax(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta)[0]
            actions.append((v, action))
            if v < alpha:
                return v, action
            beta = min(beta, v)
        return min(actions)

    def maxValue(self, state, agentIndex, depth, alpha, beta):
        actions = []
        for action in state.getLegalActions(agentIndex):
            v = self.minimax(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta)[0]
            actions.append((v, action))
            if v > beta:
                return v, action
            alpha = max(alpha, v)
        return max(actions)

    def minimax(self, state, agentIndex, depth, alpha=float('-inf'), beta=float('inf')):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state), Directions.STOP
        numAgents = state.getNumAgents()
        agentIndex %= numAgents
        if agentIndex == numAgents - 1:
            depth -= 1
        ghostState = state.getGhostState(self.index)
        scared = ghostState.scaredTimer > 0
        if agentIndex == self.index and not scared:
            return self.maxValue(state, agentIndex, depth, alpha, beta)
        else:
            return self.minValue(state, agentIndex, depth, alpha, beta)

    def evaluationFunction(self, state):
        ghostState = state.getGhostState(self.index)
        ghostPos = ghostState.getPosition()
        pacmanPos = state.getPacmanPosition()
        pacmanDis = manhattanDistance(ghostPos, pacmanPos)
        isScared = ghostState.scaredTimer > 0
        if isScared:
            ghostScore = 2 * pacmanDis
        else:
            ghostScore = 500 if pacmanDis == 0 else -2 * pacmanDis
        trappingScore = 0
        if not isScared:
            otherGhostsPos = [state.getGhostPosition(i) for i in range(1, state.getNumAgents()) if i != self.index]
            for otherGhostPos in otherGhostsPos:
                if manhattanDistance(ghostPos, otherGhostPos) < 2:
                    trappingScore -= 15
                trappingScore += max(0, 10 - manhattanDistance(otherGhostPos, pacmanPos))
        foodList = state.getFood().asList()
        foodLeft = len(foodList)
        capsules = state.getCapsules()
        capsulesLeft = len(capsules)
        foodScore = -2 * foodLeft
        capsuleScore = -20 * capsulesLeft
        score = state.getScore() + ghostScore + trappingScore + foodScore + capsuleScore
        return score


class AStarGhost(GhostAgent):
    def __init__(self, index):
        super().__init__(index)

    def getAction(self, state):
        action = self.aStarSearch(state)
        legalActions = state.getLegalActions(self.index)
        if action not in legalActions:
            action = random.choice(legalActions) if legalActions else Directions.STOP
        return action

    def aStarSearch(self, state):
        ghostPosition = state.getGhostPosition(self.index)
        pacmanPosition = state.getPacmanPosition()
        frontier = util.PriorityQueue()
        frontier.push((ghostPosition, []), 0)
        explored = set()
        while not frontier.isEmpty():
            position, path = frontier.pop()
            if position in explored:
                continue
            explored.add(position)
            if position == pacmanPosition:
                return path[0] if path else Directions.STOP
            for action in state.getLegalActions(self.index):
                successor = state.generateSuccessor(self.index, action).getGhostPosition(self.index)
                if successor not in explored:
                    new_path = path + [action]
                    cost = len(new_path) + util.manhattanDistance(successor, pacmanPosition)
                    frontier.push((successor, new_path), cost)
        return Directions.STOP
