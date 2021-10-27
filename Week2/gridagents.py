import math
import numpy
import uuid

# base class for all objects that can be in a GridWorld. Not much here other than
# the world of which they are a part and their x-y coordinates in the world (which may
# be actual or believed coordinates)


class GridObject(object):

    def __init__(self, name, obj_id=None, world=None, x=0, y=0):

        # obscure: with a __setattr__ method override, setters for any sort of internal
        # attribute must be set even in the constructor using the base class' __setattr__ method.
        # what kind of object this is
        object.__setattr__(self, "_objectName", name)
        if obj_id is None:
            # unique identifier
            object.__setattr__(self, "_objectID", uuid.uuid4().hex)
        else:
            # no special guards here against duplicate IDs - this is the user's responsibility!
            object.__setattr__(self, "_objectID", obj_id)
        # agents which are not static can take actions.
        object.__setattr__(self, "_static", False)
        self.x = x
        self.y = y
        object.__setattr__(self, "_world", world)

    # name, object ID, and world cannot be set directly. name and ID are fixed at
    # at construction. world is set through the embed method below.
    def __setattr__(self, name, value):
        if name not in ["_objectName", "_objectID", "_world", "_static"]:
            object.__setattr__(self, name, value)

    @property
    def objectName(self):
        return self._objectName

    @property
    def objectID(self):
        return self._objectID

    @property
    def inWorld(self):
        return self._world

    def embed(self, world):
        self._world = world

    def place(self, world, x, y):
        if self._world is None:
            self._world = world
        if self._world == world:
            self.x = x
            self.y = y


class Action():

    # define the possible actions here
    inaction = -1
    move = 0
    tag = 1  # new

    # set up a basic action. An action stores the agent, what action it is doing,
    # in what direction the action is made, any possible object of the action (self.actedUpon),
    # and the action start point position.
    def __init__(self, agent, code, target, direction):

        self.agent = agent
        self.actionCode = code
        self.actionDirection = direction
        self.actedUpon = target
        self.x = agent.x
        self.y = agent.y

# a GridTarget is a very simple object indeed, the only thing one
# can do to it is tag it.


class GridTarget(GridObject):

    def __init__(self, name, obj_id=None, world=None, x=0, y=0, seq=0):

        super().__init__("target", obj_id, world, x, y)
        self._tagged = False
        self._order = seq

    @property
    def isTagged(self):
        return self._tagged

    @property
    def sequenceNum(self):
        return self._order

    def tagTarget(self):
        self._tagged = True


class GridAgent(GridObject):

    # set up the agent, which needs a name, an ID, a world to live in, and a start point.
    def __init__(self, name, obj_id=None, world=None, x=0, y=0):

        # call the generic GridObject constructor to set up common properties
        super().__init__("agent", obj_id, world, x, y)
        # no current action selected
        self._currentAction = Action(self, Action.inaction, None, 0)
        self.owned = []  # any objects the agent may possess
        # a dictionary of (x,y) positions containing a target dictionary of accessible locations with distances
        self._map = {}
        # initialise our start point so we know when the map is complete
        self._frontier = [(self.x, self.y)]
        # this will keep track of what our path has been, so we can navigate back to a starting point
        self._backtrack = []
        # what should the agent's goal(s) be? This can be set either internally or externally
        self._goals = []
        self._curPath = None  # this will be the path list the agent will follow. None means nowhere to go; empty indicates at destination

    # don't allow arbitrary redirection of current actions
    def __setattr__(self, name, value):
        if name != "_currentAction":
            GridObject.__setattr__(self, name, value)

    # actionResult gives the agent its observation model: what happens when it takes an action. In general, in the GridWorld,
    # the observation will be a returned object or location indicating the agent successfully acquired the object or occupied
    # the location. If there were other agents this agent could have removed (or killed!) in its location, it can interrogate
    # the occupants property of a returned location to check that the agent in concern no longer exists.
    def actionResult(self, result):
        # filter out non-actions
        if self._currentAction.actionCode > self._currentAction.inaction:
            # move action expects a GridPoint in return. Any result observations you add should check as below for
            # the correct class!
            if self._currentAction.actionCode == self._currentAction.move:
                if result is None:
                    return
                if result.__class__.__name__ != "GridPoint":
                    raise ValueError("Expected a GridPoint class for a Move action, got a {0} class instead".format(
                        result.__class__.__name__))
                self.x = result.x
                self.y = result.y
            # any other actions you may implement should have their observed results dealt with here

            # tagging a target sets the current path to None (so we can plan another path)
            # and if the attempt to tag succeeded, removes the target from the goals. So
            # if the path reached the presumed destination but there was nothing to tag,
            # we must have gone astray, and a new path to the goal can be planned.
            if self._currentAction.actionCode == self._currentAction.tag:
                self._curPath = None
                if result is None:
                    return
                if result.__class__.__name__ != "GridTarget":
                    raise ValueError("Expected a GridTarget class for a Move action, got a {0} class instead".format(
                        result.__class__.__name__))
                self._goals.pop(0)

    # this is the main function that generates intelligent behaviour. It implements
    # a 'policy': a mapping from the state (which you can get from the world, your x, y
    # position, and the occupants which you will get as a list), to an action.
    def chooseAction(self, world, x, y, occupants):

        # don't attempt to act in a world we're not in. This also prevents us from accidentally
        # resetting the world.
        if world != self._world:
            GridObject.__setattr__(self, "_currentAction", Action(
                self, Action.inaction, None, 0))
            return self._currentAction

        # --- Insert your actions here ---

        # some goals still to reach?
        if len(self._goals) > 0:
            currentLoc = (self.x, self.y)
            # no path, so create a new one
            while self._curPath is None and len(self._goals) > 0:

                # TODO
                # ----- Choose which search method you are using with these lines -------

                self._curPath = self._depthFirstSearch(
                    currentLoc, self._goals[0])
                # self._curPath = self._breadthFirstSearch(
                #    currentLoc, self._goals[0])
                # self._curPath = self._iterativeDeepeningSearch(
                #    currentLoc, self._goals[0])
                # self._curPath = self._AStarSearch(currentLoc, self._goals[0])

                # could have an unreachable goal, which we just remove
                if self._curPath is None:
                    self._goals.pop(0)
            # no goals left; everything remaining was unreachable, so do nothing.
            if len(self._goals) == 0:
                GridObject.__setattr__(self, "_currentAction", Action(
                    self, Action.inaction, None, 0))
                return self._currentAction
            # path includes our current location, which we can just pop, leaving the path as the
            # waypoints to the next goal
            if self._curPath[0] == currentLoc:
                self._curPath.pop(0)
            # at a goal point? Current path will be empty
            if len(self._curPath) == 0:
                if currentLoc == self._goals[0]:
                    objectToTag = None
                    try:
                        # is this a taggable goal? If so, tag it.
                        objectToTag = next(
                            occ for occ in occupants if occ.objectName == "target")
                        GridObject.__setattr__(self, "_currentAction", Action(
                            self, Action.tag, objectToTag, 0))
                        return self._currentAction
                    except StopIteration:
                        pass
            else:
                # not at the goal point. Continue to move towards the next waypoint.
                GridObject.__setattr__(self, "_currentAction", Action(
                    self, Action.move, None, self._getDirection(self._curPath[0])))
                return self._currentAction

        # default action is just a random move in some direction.
        GridObject.__setattr__(self, "_currentAction", Action(
            self, Action.move, None, round(numpy.random.uniform(-0.49999, 3.5))))
        return self._currentAction

    # importMap updates the world map, either in whole or in part.
    def importMap(self, gridMap):
        self._map.update(gridMap)

    # addGoalPoint indicates that this x, y position is to be considered a goal point. Priority
    # allows the point to be inserted wherever desired in the list.
    def addGoalPoint(self, x, y, priority=-1):
        # some points may be reachable, but not in the map because they have been optimised
        # away. We can exploit the geometries of the GridWorld to derive these points, because
        # a valid goal MUST lie between 2 points which are in the map, which share either an
        # x or y coordinate, and which are connected
        if (x, y) not in self._map:
            # first, find the set of points that share an x or y coordinate with the goal
            alignedPoints = sorted(
                [loc for loc in self._map if loc[0] == x or loc[1] == y])
            # now, see if one lies beyond the goal in the x or y direction
            nextInColumn = None
            nextInRow = None
            try:
                # a Python generator function extracts the first point with the same
                # x-coordinate lying below (i.e. has a larger y-value) than the goal.
                nextInColumn = next(origin for origin in sorted(
                    alignedPoints, key=lambda l: l[1]) if origin[1] > y)
            except StopIteration:
                pass
            try:
                # same idea for the a point with the same y coordinate lying to the right
                # of the goal
                nextInRow = next(
                    origin2 for origin2 in alignedPoints if origin2[0] > x)
            except StopIteration:
                pass

            if nextInColumn is not None:
                # a column-aligned point was found. Does it have a connection (an edge) to a
                # neighbouring point lying above the goal?
                previousInColumn = None
                try:
                    previousInColumn = next(dst for dst in sorted([loc3 for loc3 in self._map[nextInColumn]
                                                                   if loc3[0] == x and loc3[1] < y],
                                                                  key=lambda m: m[1], reverse=True))
                except StopIteration:
                    pass
                # goal is between 2 connected points in a column. Insert the goal point as a node,
                # and replace the single edge between the original 2 nodes with 2 edges linking the
                # goal point.
                if previousInColumn is not None:
                    self._map[(x, y)] = {
                        nextInColumn: nextInColumn[1]-y, previousInColumn: y-previousInColumn[1]}
                    self._map[previousInColumn][(x, y)] = y-previousInColumn[1]
                    self._map[nextInColumn][(x, y)] = nextInColumn[1]-y
                    del self._map[previousInColumn][nextInColumn]
                    del self._map[nextInColumn][previousInColumn]

            if nextInRow is not None:
                # same logic, for row-aligned points
                previousInRow = None
                try:
                    previousInRow = next(dst2 for dst2 in sorted([loc4 for loc4 in self._map[nextInRow]
                                                                  if loc4[1] == y and loc4[0] < x],
                                                                 reverse=True))
                except StopIteration:
                    pass
                if previousInRow is not None:
                    self._map[(x, y)] = {
                        nextInRow: nextInRow[0]-x, previousInRow: x-previousInRow[0]}
                    self._map[previousInRow][(x, y)] = x-previousInRow[0]
                    self._map[nextInRow][(x, y)] = nextInRow[0]-x
                    del self._map[previousInRow][nextInRow]
                    del self._map[nextInRow][previousInRow]
        # so now if the goal is still on the map, it is fundamentally unreachable.
        if (x, y) not in self._map:
            raise ValueError("Can't get there from here! Specfied a goal point ({0},{1}) not reachable in agent {2}'s map".format(
                x, y, self.objectID))
            return
        # reachable points can be inserted at their appropriate point in the target list.
        if priority < 0:
            self._goals.append((x, y))
        else:
            self._goals.insert(priority, (x, y))

    # get rid of an existing goal point
    def removeGoalPoint(self, x, y):
        try:
            self._goals.remove(x, y)
        except ValueError:
            pass

    # convenience function allows us to extract the direction to a target location
    def _getDirection(self, target, immediate_neighbours_only=True):
        # [2021-10-27 SM] changed to adjacent-only cells
        if immediate_neighbours_only:
            if target[0] == self.x and target[1] == self.y - 1:
                return self._world.North
            elif target[0] == self.x + 1 and target[1] == self.y:
                return self._world.East
            elif target[0] == self.x and target[1] == self.y + 1:
                return self._world.South
            elif target[0] == self.x - 1 and target[1] == self.y:
                return self._world.West
            else:
                return self._world.Nowhere
        else:
            # Get direction - from any distance
            # TODO
            return self._world.Nowhere

    # an efficient way to identify if a tuple is in a list. Creates a python generator expression to evaluate.
    def _inFrontier(self, target):
        # deprecated
        return self._inTupleList(target, self._frontier)

    def _inTupleList(self, target, list):
        try:
            nextTgt = next(
                loc for loc in list if loc[0] == target[0] and loc[1] == target[1])
        except StopIteration:
            return None
        return nextTgt

    def _depthFirstSearch_original(self, start, target, ply=0, explored=None):
        # Problems with depthFirst:
        # -It will find the FIRST result it can, not the most optimal.
        # TODO find out if we want t o

        x = start[0]
        y = start[1]
        print("_depthFirstSearch_original at ({0}, {1})".format(x, y))

        if (x, y) not in self._map:
            self._map[(x, y)] = {}
        if (x, y) in self._frontier:
            self._frontier.remove((x, y))
        if explored is None:
            explored = []
        if start == target:
            return explored
        if ply > 12:
            print("leaving depthfirst - ply max of 12 reached")
            return

        result = None
        self._backtrack.append(start)
        here = self._world.getLocation(x, y)
        if here.canGo(self._world.North):
            nextLoc = (x, y - 1)
            if nextLoc not in explored:
                explored.append(nextLoc)
                result = self._depthFirstSearch(
                    nextLoc, target, ply + 1, explored)
            if result is not None:
                return result
        if here.canGo(self._world.East):
            nextLoc = (x + 1, y)
            if nextLoc not in explored:
                explored.append(nextLoc)
                result = self._depthFirstSearch(
                    nextLoc, target, ply + 1, explored)
            if result is not None:
                return result
        if here.canGo(self._world.South):
            nextLoc = (x, y + 1)
            if nextLoc not in explored:
                explored.append(nextLoc)
                result = self._depthFirstSearch(
                    nextLoc, target, ply + 1, explored)
            if result is not None:
                return result
        if here.canGo(self._world.West):
            nextLoc = (x - 1, y)
            if nextLoc not in explored:
                explored.append(nextLoc)
                result = self._depthFirstSearch(
                    nextLoc, target, ply + 1, explored)
            if result is not None:
                return result

        # start = ???
        # target = ???
        # ply = the thing with the thing
        # fff
        # NESW - North is node 1. First traversal should look like NNNNNNNNNN^ENNN (etc etc)
        return None

    def _depthFirstSearch(self, start, target, ply=0, explored=None):
        x = start[0]
        y = start[1]
        # print("_depthFirstSearch at ({0}, {1})".format(x, y))

        if (x, y) not in self._map:
            self._map[(x, y)] = {}
        if (x, y) in self._frontier:
            self._frontier.remove((x, y))
        if explored is None:
            explored = []
        if start == target:
            return explored
        if ply > 9:
            # print("leaving depthfirst - ply max of 12 reached")
            return

        result = None
        result_north = None
        result_east = None
        result_south = None
        result_west = None
        self._backtrack.append(start)
        here = self._world.getLocation(x, y)
        if here.canGo(self._world.North):
            nextLoc = (x, y - 1)
            if nextLoc not in explored:
                result_north = self._depthFirstSearch(
                    nextLoc, target, ply + 1, explored + [nextLoc])
        if here.canGo(self._world.East):
            nextLoc = (x + 1, y)
            if nextLoc not in explored:
                result_east = self._depthFirstSearch(
                    nextLoc, target, ply + 1, explored + [nextLoc])
        if here.canGo(self._world.South):
            nextLoc = (x, y + 1)
            if nextLoc not in explored:
                result_south = self._depthFirstSearch(
                    nextLoc, target, ply + 1, explored + [nextLoc])
        if here.canGo(self._world.West):
            nextLoc = (x - 1, y)
            if nextLoc not in explored:
                result_west = self._depthFirstSearch(
                    nextLoc, target, ply + 1, explored + [nextLoc])

        largest_distance = float('inf')

        if result_north is not None:
            if largest_distance > len(result_north):
                largest_distance = len(result_north)
                result = result_north
        if result_east is not None:
            if largest_distance > len(result_east):
                largest_distance = len(result_east)
                result = result_east
        if result_south is not None:
            if largest_distance > len(result_south):
                largest_distance = len(result_south)
                result = result_south
        if result_west is not None:
            if largest_distance > len(result_west):
                largest_distance = len(result_west)
                result = result_west

        tabs = " " * ply
        # print("!\n{0}{1}\n{2}{3}\n{4}{5}\n{6}{7}".format(tabs, distance_north,
        #      tabs, distance_east, tabs, distance_south, tabs, distance_west))
        #print(tabs + "traversed " + traversed)

        return result

    # TODO
    # breadth-first search should expand each location completely before moving to the next. In the
    # gridworld, this isn't crippling, the branching factor is only 4, but consider how the problem
    # would scale to a 100*100 grid (!)

    def _breadthFirstSearch(self, start, target):
        x = start[0]
        y = start[1]
        print("_breadthFirstSearch at ({0}, {1})".format(x, y))

        return None

    # TODO
    # here is the extension to iterative deepening
    def _iterativeDeepeningSearch(self, start, target):
        x = start[0]
        y = start[1]
        print("depthfirst at ({0}, {1})".format(x, y))

        return None

    # TODO
    # A* search is an informed search, and expects a heuristic, which should be a
    # function of 2 variables, both tuples, the start, and the target.
    def _AStarSearch(self, start, target, heuristic=None):
        x = start[0]
        y = start[1]
        print("depthfirst at ({0}, {1})".format(x, y))

        return None
