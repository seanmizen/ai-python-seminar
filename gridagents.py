import math
import numpy
import uuid

# base class for all objects that can be in a GridWorld. Not much here other than
# the world of which they are a part and their x-y coordinates in the world (which may
# be actual or believed coordinates)
# test


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

        # TODO
        # --- Insert your actions here ---
        # Needed: explore the world. Decide if there's still something to explore, then
        # use appropriate search to explore the space.

        # If the frontier exists at all, the map is not yet completely built and compacted
        if self._frontier is not None:

            # still some places to explore?
            if len(self._frontier) > 0:
                return self._depthFirstExploration(world, x, y)
            # the map is finished but we still need to prune
            else:
                self._pruneMap()

        # default action is just a random move in some direction.
        GridObject.__setattr__(self, "_currentAction", Action(
            self, Action.move, None, round(numpy.random.uniform(-0.49999, 3.5))))
        return self._currentAction

    # TODO
    # implement depth-first search, creating a map in the process
    # a depth-first exploration should proceed as far as it can, by choosing a direction at each point
    # where a decision is possible, then 'backtracking' once no further choices are available, back
    # to the last point where a choice was possible.

    def _depthFirstExploration(self, world, x, y):
        # get the next place to check. Are we there?
        if x == self._frontier[-1][0] and y == self._frontier[-1][1]:
            # Yes. Explore.
            nowAt = self._frontier.pop()
            # add our point to the map
            self._map[nowAt] = {}
            # repetitive, but the indexing makes this more obvious than a for-loop:
            # see what other locations we can access directly from here.
            here = world.getLocation(x, y)
            goingPlaces = False
            if here.canGo(world.North):
                there = (nowAt[0], nowAt[1]-1)
                if there not in self._map and self._inFrontier(there) is None:
                    goingPlaces = True
                    self._frontier.append(there)
                self._map[nowAt][there] = 1
            if here.canGo(world.East):
                there = (nowAt[0]+1, nowAt[1])
                if there not in self._map and self._inFrontier(there) is None:
                    goingPlaces = True
                    self._frontier.append(there)
                self._map[nowAt][there] = 1
            if here.canGo(world.South):
                there = (nowAt[0], nowAt[1]+1)
                if there not in self._map and self._inFrontier(there) is None:
                    goingPlaces = True
                    self._frontier.append(there)
                self._map[nowAt][there] = 1
            if here.canGo(world.West):
                there = (nowAt[0]-1, nowAt[1])
                if there not in self._map and self._inFrontier(there) is None:
                    goingPlaces = True
                    self._frontier.append(there)
                self._map[nowAt][there] = 1
            # somewhere new to move to - go to it
            if goingPlaces:
                self._backtrack.append(nowAt)
                GridObject.__setattr__(self, "_currentAction", Action(
                    self, Action.move, None, self._getDirection(self._frontier[-1])))
            # nowhere new: backtrack to our previous position
            elif len(self._backtrack) > 0:
                GridObject.__setattr__(self, "_currentAction", Action(
                    self, Action.move, None, self._getDirection(self._backtrack.pop())))
            # nowhere at all: we are painted into a corner; no point in trying to move.
            else:
                GridObject.__setattr__(self, "_currentAction", Action(
                    self, Action.inaction, None, -1))
            return self._currentAction
        # not there. We must have backtracked.
        elif (x, y) in self._map:
            # Has this depth of the backtrack chain been thoroughly explored?
            if self._frontier[-1] in self._map[(x, y)]:
                # No. Go down the next available path
                self._backtrack.append((x, y))
                GridObject.__setattr__(self, "_currentAction", Action(
                    self, Action.move, None, self._getDirection(self._frontier[-1])))
            # Yes. Backtrack another step if we can
            elif len(self._backtrack) > 0:
                GridObject.__setattr__(self, "_currentAction", Action(
                    self, Action.move, None, self._getDirection(self._backtrack.pop())))
            else:
                raise RuntimeError("Backtracked into a brick wall whilst exploring! Expected point ({0},{1}) unreachable".format(
                    self._frontier[-1][0], self._frontier[-1][1]))
                GridObject.__setattr__(self, "_currentAction", Action(
                    self, Action.inaction, None, -1))
            return self._currentAction
        raise RuntimeError(
            "Ran off the edge of the map! No map location exists for ({0},{1})".format(x, y))
        GridObject.__setattr__(self, "_currentAction",
                               Action(self, Action.inaction, None, -1))
        return self._currentAction

    # TODO
    # prune the map to get rid of uninteresting 'corridor' points where no turns are allowed
    def _pruneMap(self):
        locsToDelete = []  # list of map locations that can be pruned
        # FIXME this just exits
        while self._frontier is not None:
            self._frontier = None

    # convenience function allows us to extract the direction to a target location
    def _getDirection(self, target):
        if target[0] == self.x:
            if target[1] == self.y:
                return self._world.Nowhere
            elif target[1] > self.y:
                return self._world.South
            else:
                return self._world.North
        elif target[0] < self.x:
            if target[1] != self.y:
                return self._world.Nowhere
            else:
                return self._world.West
        else:
            if target[1] != self.y:
                return self._world.Nowhere
            else:
                return self._world.East

    # an efficient way to identify if a tuple is in a list. Creates a python generator expression to evaluate.
    def _inFrontier(self, target):
        try:
            nextTgt = next(
                loc for loc in self._frontier if loc[0] == target[0] and loc[1] == target[1])
        except StopIteration:
            return None
        return nextTgt
