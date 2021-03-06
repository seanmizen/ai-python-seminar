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

        if len(self._frontier) > 0:
            GridObject.__setattr__(
                self, "_currentAction", self._depthFirstExploration(world, x, y))
        else:
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
        # FIXME build the map, indexed by (origin)(destination) pairs
        # self._map[(x, y)] = {}
        # self._map[(x, y)][(x+1, y+1)] = {}
        # _map[start][end] = distance (1)
        # if possible_location in _map[my_location]:

        if (x, y) not in self._map:
            self._map[(x, y)] = {}
        if (x, y) in self._frontier:
            self._frontier.remove((x, y))

        # step 1: can I go NESW? add to frontier.
        # current_location is an instance of GridPoint (see: gridworld)
        current_gridpoint = world.getLocation(x, y)
        if current_gridpoint.canGo(world.North):
            next_location = (x, y - 1)
            if next_location not in self._map and next_location not in self._frontier:
                self._frontier.append(next_location)
            self._map[(x, y)][next_location] = 1
        if current_gridpoint.canGo(world.East):
            next_location = (x + 1, y)
            if next_location not in self._map and next_location not in self._frontier:
                self._frontier.append(next_location)
            self._map[(x, y)][next_location] = 1
        if current_gridpoint.canGo(world.South):
            next_location = (x, y + 1)
            if next_location not in self._map and next_location not in self._frontier:
                self._frontier.append(next_location)
            self._map[(x, y)][next_location] = 1
        if current_gridpoint.canGo(world.West):
            next_location = (x - 1, y)
            if next_location not in self._map and next_location not in self._frontier:
                self._frontier.append(next_location)
            self._map[(x, y)][next_location] = 1

        # step 2: now I've explored the area around me, choose a next step.
        if len(self._frontier) > 0:
            if current_gridpoint.canGo(self._getDirection(self._frontier[-1])):
                # continue depth search
                self._backtrack.append((x, y))
                GridObject.__setattr__(self, "_currentAction", Action(
                    self, Action.move, None, self._getDirection(self._frontier.pop())))
            else:
                # backtrack
                if len(self._backtrack) > 0:
                    GridObject.__setattr__(self, "_currentAction", Action(
                        self, Action.move, None, self._getDirection(self._backtrack.pop())))
                else:
                    # nowhere to go. No backtrack, no frontier.
                    raise()
        else:
            GridObject.__setattr__(self, "_currentAction", Action(
                self, Action.inaction, None, 0))

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
        # [2021-10-27 SM] changed to adjacent-only cells
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

    # an efficient way to identify if a tuple is in a list. Creates a python generator expression to evaluate.
    def _inFrontier(self, target):
        # deprecated
        return self.inTupleList(target, self._frontier)

    def inTupleList(self, target, list):
        try:
            nextTgt = next(
                loc for loc in list if loc[0] == target[0] and loc[1] == target[1])
        except StopIteration:
            return None
        return nextTgt
