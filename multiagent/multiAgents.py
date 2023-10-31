# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        """PAC man has two states (drugged up and normal), position
        Ghosts have Two Stats (killer or scared), a timer, and position 
        Game states is the score, 
        States we need to track: 
        1) PACs position 
        2) PACs drug level 1 or none 
        3) Ghost position 
        4) Ghost State Scared or not 
        5) Ghosts timer
        6) Games Scores 
        7) Game Food Postions"""

        #Address each state by priority

        #Ghost position
        # Calculate the distance to the closest ghost
        ghostDistances = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        closestGhostDistance = min(ghostDistances)
        # two is used not one since one causes interaction with killer ghost
        if closestGhostDistance < 2:
            return -float('inf')

        # food Count and Positions
        # NO killer ghost so eat up Pman
        closestFoodDistance = float('inf')
        foodCount = currentGameState.getNumFood()
        nextFoodCount = successorGameState.getNumFood()
        if nextFoodCount < foodCount:
            return float('inf')

        # This pioritise food as the 2nd highest after ghosts
        # Food in list form from method notes
        # Calculate the distance to the closest food pellet
        foodDistances = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
        closestFoodDistance = min(foodDistances)

        # we must return the inverse to force node attempt
        return 1.0 / closestFoodDistance

        # given form project
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def minimax(self, depth, agentIndex, gameState):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        # Max layer (Pacman)
        if agentIndex == 0:
            bestValue = float('-inf')
            for move in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, move)
                # Next agent is a min agent
                value = self.minimax(depth, agentIndex + 1, successor)
                bestValue = max(bestValue, value)
            return bestValue
        # Min layer (Ghosts)
        else:
            bestValue = float('inf')
            for move in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, move)
                if agentIndex == gameState.getNumAgents() - 1:
                    # Switch to the max layer (Pacman)
                    value = self.minimax(depth - 1, 0, successor)
                else:
                    # Stay in the min layer (Ghosts)
                    value = self.minimax(depth, agentIndex + 1, successor)
                bestValue = min(bestValue, value)
            return bestValue

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Pacman's actions
        legalActions = gameState.getLegalActions(0)
        bestValue = float('-inf')
        bestAction = None
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            # Start with min layer (ghosts)
            value = self.minimax(self.depth, 1, successor)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

        #given as part of project
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabeta(self, depth, agentIndex, gameState, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            # evaluation function for leaf nodes
            return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(agentIndex)

        # Check if it's Pacman's turn (max layer)
        if agentIndex == 0:
            bestValue = float('-inf')
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                value = self.alphabeta(depth, agentIndex + 1, successor, alpha, beta)
                bestValue = max(bestValue, value)

                # Alpha-beta pruning
                if bestValue >= beta:
                    return bestValue

                alpha = max(alpha, bestValue)
            return bestValue
        # Min layer (Ghosts)
        else:
            bestValue = float('inf')
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)

                if agentIndex == gameState.getNumAgents() - 1:
                    value = self.alphabeta(depth - 1, 0, successor, alpha, beta)
                else:
                    value = self.alphabeta(depth, agentIndex + 1, successor, alpha, beta)

                bestValue = min(bestValue, value)

                # Alpha-beta pruning
                if bestValue <= alpha:
                    return bestValue

                beta = min(beta, bestValue)
            return bestValue

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        bestValue = float('-inf')

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = self.alphabeta(self.depth, 1, successor, alpha, beta)

            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction
        #given at project start
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, state, depth, agentIndex):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if agentIndex == 0:  # Pacman's turn (max layer)
            maxAction = None
            maxUtility = float('-inf')
            for action in state.getLegalActions(0):
                successorState = state.generateSuccessor(0, action)
                successorUtility = self.expectimax(successorState, depth - 1, 1)
                if successorUtility > maxUtility:
                    maxUtility = successorUtility
                    maxAction = action
            return maxUtility

        else:  # Ghost's turn (expectation layer)
            legalActions = state.getLegalActions(agentIndex)
            expectedUtility = 0
            numActions = len(legalActions)
            for action in legalActions:
                successorState = state.generateSuccessor(agentIndex, action)
                successorUtility = self.expectimax(successorState, depth, (agentIndex + 1) % state.getNumAgents())
                expectedUtility += successorUtility / numActions
            return expectedUtility

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestUtility = float('-inf')

        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            successorUtility = self.expectimax(successorState, self.depth, 1)
            if successorUtility > bestUtility:
                bestUtility = successorUtility
                bestAction = action

        return bestAction
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: We want to create a more effective evaluation function that
    takes into account various factors to help Pacman make better decisions.
    The factors we consider are:
    1. Game score: Higher score is better.
    2. Remaining food: Less remaining food is better.
    3. Distance to the nearest ghost: Avoid ghosts.
    4. Distance to the nearest scared ghost: Seek out scared ghosts.

    We assign weights to each factor and combine them to compute the evaluation
    score for a state.
    """
    "*** YOUR CODE HERE ***"
    # Extract game state information
    pacmanPosition = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Define weights for the factors
    scoreWeight = 100  # Weight for game score
    foodWeight = -1  # Weight for remaining food (negative because we want less remaining)
    ghostDistanceWeight = -10  # Weight for distance to the nearest ghost (negative because we want to avoid them)
    scaredGhostDistanceWeight = 5  # Weight for distance to the nearest scared ghost (positive because we want to catch them)

    # Calculate the distance to the nearest ghost
    nearestGhostDistance = min(
        manhattanDistance(pacmanPosition, ghostState.getPosition()) for ghostState in ghostStates)

    # Calculate the distance to the nearest scared ghost
    nearestScaredGhostDistance = min(manhattanDistance(pacmanPosition, ghostState.getPosition())
                                     for ghostState, scaredTime in zip(ghostStates, scaredTimes) if scaredTime > 0)

    # Calculate the evaluation score
    evaluationScore = (currentGameState.getScore() * scoreWeight +
                       foodGrid.count() * foodWeight +
                       nearestGhostDistance * ghostDistanceWeight +
                       nearestScaredGhostDistance * scaredGhostDistanceWeight)

    return evaluationScore
    #given
    #util.raiseNotDefined()



# Abbreviation
better = betterEvaluationFunction
