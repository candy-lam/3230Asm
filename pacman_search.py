# search.py
# ---------
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

# If you have any question, ask in Piazza or email me (mxchen21@cse.cuhk.edu.hk)
# DO NOT copy the answer from any website. You can refer to tutorial slides, but try it by yourself first! 

"""
CSCI3230 / ESTR3108 2021-22 First Term Assignment 4 

I declare that the assignment here submitted is original except for source material explicitly acknowledged, 
and that the same or closely related material has not been previously submitted for another course. 
I also acknowledge that I am aware of University policy and regulations on honesty in academic work, 
and of the disciplinary guidelines and procedures applicable to breaches of such policy and regulations, 
as contained in the following websites. 

University Guideline on Academic Honesty: 
http://www.cuhk.edu.hk/policy/academichonesty/ 
Faculty of Engineering Guidelines to Academic Honesty:
http://www.erg.cuhk.edu.hk/erg-intra/upload/documents/ENGG_Discipline.pdf 

Student Name: Lam Hiu Ching
Student ID : 1155129247
"""

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from sys import path
import util
import heapq

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """ 
    Question 1: Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    You only need to submit this file. Do not change other files!
    If you finish this function, you almost finish all the questions!
    Read util.py to find suitable data structure!
    All you need is pass all the code in commands.txt
    """

    # SOLUTION 1 iterative function
    visited_list = [] #initilize empty visited list
    fringe = util.Stack() 
    fringe.push((problem.getStartState(), []))  #push start state into the stack
    while (fringe): 
        object = fringe.pop()
        node = object[0]
        move = object[1]     
        if (node not in visited_list):
            visited_list.append(node)
            if (problem.isGoalState(node)): return move     #goal then return
            for success in reversed(problem.getSuccessors(node)): #success[0]: coordinate [1]:direction [2]:cost
                if (success not in visited_list):                    
                    fringe.push((success[0], (move + [success[1]])))
    return []





def breadthFirstSearch(problem):
    """Question 2: Search the shallowest nodes in the search tree first."""
    visited_list = []
    fringe = util.Queue()
    fringe.push((problem.getStartState(), []))
    while (fringe):
        object = fringe.pop()
        node = object[0]
        move = object[1]
        if (node not in visited_list):
            if (problem.isGoalState(node)): return move     #goal then return 
            visited_list.append(node)
            for success in problem.getSuccessors(node): #success[0]:coordinate [1]:direction [2]:cost
                if (success not in visited_list):
                    fringe.push((success[0], (move + [success[1]])))
    return []


def uniformCostSearch(problem):
    """Question 3: Search the node of least total cost first."""
    visited_list = []
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), []), problem)
    while (fringe):
        object = fringe.pop()
        node = object[0]
        move = object[1]  
        if (node not in visited_list):
            if (problem.isGoalState(node)): return move
            for success in problem.getSuccessors(node): #success[0]:coordinate [1]:direction [2]:cost
                if (success not in visited_list):
                    next = move + [success[1]]
                    ncost = problem.getCostOfActions(next)
                    visited_list.append(node)
                    fringe.push((success[0], next), ncost)  #push and sort by ncost
    return []



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Question 4: Search the node that has the lowest combined cost and heuristic first."""
    visited_list = []
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), []), heuristic(problem.getStartState(), problem))
    while (fringe):
        object = fringe.pop()
        node = object[0]
        move = object[1]  
        if (node not in visited_list):            
            if (problem.isGoalState(node)): return move
            for success in problem.getSuccessors(node): #success[0]:coordinate [1]:direction [2]:cost
                if (success not in visited_list):
                    next = move + [success[1]]
                    ncost = problem.getCostOfActions(next) + heuristic(success[0], problem)
                    visited_list.append(node)
                    fringe.push((success[0], next), ncost)  #push and sort by ncost
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
