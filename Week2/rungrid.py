import sys
import gridworld
import gridagents
import pygame
import threading
import time

# create objects for the GridWorld

# we enclose the whole world in its own separate function so it can run in a thread.
# that lets us run the graphics updating and redraws separately and speeds performance.
# (not that ultimate performance here is really the goal).
# sizeX and sizeY are, obviously, the dimensions of the gridworld, time is the runtime,
# worldInfo is a dictionary that lets you pass in your own parameters and receive back
# whatever information you might want to dump to file or draw on screen or whatever.


def runWorld(sizeX, sizeY, time, stop, worldInfo):

    threadRunTime = time
    threadTime = 0
    wargs = {}

    # check for input arguments that set up the environment
    if 'points' in worldInfo:
        wargs['points'] = worldInfo['points']
    if 'occupants' in worldInfo:
        wargs['occupants'] = worldInfo['occupants']
    world = gridworld.GridWorld(sizeX, sizeY, time, **wargs)
    if 'targets' in worldInfo:
        seq = 0
        targets = []
        for target in worldInfo['targets']:
            targetX = (gridagents.GridTarget(name="targetX",
                       world=world, x=target[0], y=target[1], seq=seq))
            world.placeOccupant(targetX, targetX.y, targetX.x)
            targets.append(targetX)
            seq += 1
    if 'agent1Pos' in worldInfo:
        agent1 = gridagents.GridAgent(
            name="agent1", world=world, x=worldInfo['agent1Pos'][0], y=worldInfo['agent1Pos'][1])
    else:
        agent1 = gridagents.GridAgent(
            name="agent1", world=world, x=round(sizeX/2), y=round(sizeY/2))
    world.placeOccupant(agent1, agent1.y, agent1.x, True)

    # this loop drives the simulation forward one step at a time and outputs the agent's
    # position and the time.
    while threadTime < threadRunTime:

        # exit if 'q' has been pressed
        if stop.is_set():
            threadRunTime = 0
        else:
            world.run(1)
            if threadTime != world.time:
                worldInfo['time'] = world.time
                worldInfo['agent1Pos'] = (agent1.x, agent1.y)
                for target in range(len(targets)):
                    worldInfo['targets'][target][2] = targets[target].isTagged
                threadTime += 1

# create the GridWorld itself


worldX = 10
worldY = 10
runtime = 200
displaySize = (800, 600)  # this can be changed for larger size if desired
targetsOn = True

# set up some simple 'maze' environments
points1 = dict([((x, y), 1) for x in range(worldX) for y in range(worldY)])
points2 = dict([((x, y), 1) for x in range(worldX) for y in range(worldY)])
points3 = dict([((x, y), 1) for x in range(worldX) for y in range(worldY)])
points4 = dict([((x, y), 1) for x in range(worldX) for y in range(worldY)])
# these are the walls for world 2
blanks1 = dict([((2, y), 0) for y in range(2, 8)])
blanks2 = dict([((x, 4), 0) for x in range(4, 7)])
blanks3 = dict([((8, y), 0) for y in range(2, 8)])
# these are the walls for world 3
blanks4 = dict([((x, 5), 0) for x in range(0, 4)])
blanks5 = dict([((3, y), 0) for y in range(0, 3)])
blanks6 = dict([((5, y), 0) for y in range(7, 10)])
blanks7 = dict([((7, y), 0) for y in range(0, 2)])
blanks8 = dict([((x, 4), 0) for x in range(7, 10)])
# and these are the walls for world 4. World4 is a genuine maze
# with random walls, so it's easier to specify it as a set of points
# than as continuous walls
blanks9 = {(2, 0): 0, (7, 0): 0,
           (2, 2): 0, (3, 2): 0, (5, 2): 0, (7, 2): 0,
           (2, 3): 0, (3, 3): 0, (5, 3): 0, (6, 3): 0, (7, 3): 0, (9, 3): 0,
           (6, 4): 0, (7, 4): 0, (9, 4): 0,
           (2, 5): 0, (3, 5): 0, (9, 5): 0,
           (0, 6): 0, (2, 6): 0, (4, 6): 0, (6, 6): 0, (7, 6): 0, (8, 6): 0, (9, 6): 0,
           (4, 7): 0, (6, 7): 0,
           (3, 8): 0, (4, 8): 0, (6, 8): 0}

# insert the walls into their respective environments
points2.update(blanks1)
points2.update(blanks2)
points2.update(blanks3)
points3.update(blanks4)
points3.update(blanks5)
points3.update(blanks6)
points3.update(blanks7)
points3.update(blanks8)
points4.update(blanks9)

# set up either 1 or 3 targets
# you can create more targets if you like - the format is [x, y, isVisitor] where isVisitor says
# whether the target counts as a 'visitor' - a transient occupant of the point (x, y). isVisitor = False
# means the target permanently occupies the point.
target1 = [[9, 0, False]]
target3 = [[9, 0, False], [0, 0, False], [9, 9, False]]

# choose which targets you want to aim for
targets = target1
targetPoints = dict([((target[0], target[1]), 2) for target in targets])
if targetsOn:
    points1.update(targetPoints)
    points2.update(targetPoints)
    points3.update(targetPoints)
    points4.update(targetPoints)

# and initialise the values that will be passed into their worlds. World 2 gets
# a different start point because otherwise the agent would be rather unnecessarily
# close to a wall.
worldValues1 = {'time': 0, 'agent1Pos': (
    round(worldX/2), round(worldY/2)), 'points': points1}
worldValues2 = {'time': 0, 'agent1Pos': (5, 7), 'points': points2}
worldValues3 = {'time': 0, 'agent1Pos': (
    round(worldX/2), round(worldY/2)), 'points': points3}
worldValues4 = {'time': 0, 'agent1Pos': (
    round(worldX/2), round(worldY/2)), 'points': points4}

# choose which world you want to be in here. You can also make your own
worldValues = worldValues3
if targetsOn:
    worldValues['targets'] = targets

# set up the GUI. Pygame is rather low-level, so this all looks like very 'bit-twiddling' commands
pygame.init()
# |pygame.SCALED arrgh...new in pygame 2.0, but pip install installs 1.9.6 on Ubuntu 16.04 LTS
displaySurface = pygame.display.set_mode(
    size=displaySize, flags=pygame.RESIZABLE)
aspectRatio = worldX/worldY
if aspectRatio > 4/3:
    surfaceSize = (700, 700/aspectRatio)
else:
    surfaceSize = (aspectRatio*500, 500)
displayedBackground = pygame.Surface(surfaceSize)
displayedBackground.fill(pygame.Color(224, 224, 255))
backgroundRect = pygame.Rect(round((displaySize[0]-surfaceSize[0])/2), round(
    (displaySize[1]-surfaceSize[1])/2), surfaceSize[0], surfaceSize[1])
squares = [[pygame.Rect(round(x*surfaceSize[0]/worldX), round(y*surfaceSize[1]/worldY), round(
    surfaceSize[0]/worldX), round(surfaceSize[1]/worldY)) for y in range(worldY)] for x in range(worldX)]
gridSquares = [[pygame.Surface((round(surfaceSize[0]/worldX), round(surfaceSize[1]/worldY)))
                for y in range(worldY)] for x in range(worldX)]

# A comprehension here would be more elegant. Unfortunately pygame doesn't appear
# to provide an interface for this. The pygame model seems to assume a fairly
# sequential, incremental approach to drawing
for x in range(len(gridSquares)):
    for y in range(len(gridSquares[x])):
        if 'points' in worldValues:
            if worldValues['points'][(x, y)] >= 1:
                gridSquares[x][y].fill(pygame.Color(224, 224, 255))
            else:
                gridSquares[x][y].fill(pygame.Color(0, 0, 255))
        else:
            gridSquares[x][y].fill(pygame.Color(224, 224, 255))
        # note that the rectangle target in draw.rect refers to a Rect relative to the source surface, not an
        # absolute-coordinates Rect.
        pygame.draw.rect(gridSquares[x][y], pygame.Color(
            0, 0, 0), squares[0][0], 5)

# blits transfer surfaces to another
displayedBackground.blits([(gridSquares[x][y], squares[x][y]) for x in range(
    len(gridSquares)) for y in range(len(gridSquares[0]))])
displaySurface.blit(displayedBackground, backgroundRect)

# and this obscurely-named command is the one that *actually* redraws the screen.
pygame.display.flip()

# set up the thread that actually runs the simulation
globalExit = threading.Event()
worldThread = threading.Thread(target=runWorld, name='worldThread', kwargs={
                               'sizeX': worldX, 'sizeY': worldY, 'time': runtime, 'stop': globalExit, 'worldInfo': worldValues})
if 'agent1Pos' in worldValues:
    agentOldPos = [worldValues['agent1Pos'][0], worldValues['agent1Pos'][1]]
else:
    agentOldPos = [round(worldX/2), round(worldY/2)]
curTime = 0

# draw the agent in its initial position
pygame.draw.circle(gridSquares[agentOldPos[0]][agentOldPos[1]], pygame.Color(255, 0, 0), (round(
    surfaceSize[0]/(2*worldX)), round(surfaceSize[1]/(2*worldY))), round(surfaceSize[0]/(3*worldX)))
displayedBackground.blit(
    gridSquares[agentOldPos[0]][agentOldPos[1]], squares[agentOldPos[0]][agentOldPos[1]])
displaySurface.blit(displayedBackground, backgroundRect)
pygame.display.flip()

# draw the targets in their initial position
for target in worldValues['targets']:
    pygame.draw.circle(gridSquares[target[0]][target[1]], pygame.Color(0, 128, 128), (round(
        surfaceSize[0]/(2*worldX)), round(surfaceSize[1]/(2*worldY))), round(surfaceSize[0]/(4*worldX)))
displayedBackground.blits([(gridSquares[target[0]][target[1]],
                          squares[target[0]][target[1]]) for target in worldValues['targets']])


# start the simulation
worldThread.start()

# keep redrawing until the end of the simulation
while curTime < runtime:

    # you can end the simulation by pressing 'q'. This triggers an event which is also passed into the world loop
    try:
        quitevent = next(evt for evt in pygame.event.get(
        ) if evt.type == pygame.KEYDOWN and evt.key == pygame.K_q)
        globalExit.set()
        pygame.quit()
        sys.exit()
    except StopIteration:
        redraw = False
        if curTime != worldValues['time']:
            # redraw the agent's position on the screen, which also requires redrawing the square it has come from.
            # some race conditions here, given that the agent lives in a GridWorld updated in a separate thread, but that
            # might only result in some odd draws.
            if worldValues['agent1Pos'][0] != agentOldPos[0] or worldValues['agent1Pos'][1] != agentOldPos[1]:
                pygame.draw.circle(gridSquares[worldValues['agent1Pos'][0]][worldValues['agent1Pos'][1]], pygame.Color(
                    255, 0, 0), (round(surfaceSize[0]/(2*worldX)), round(surfaceSize[1]/(2*worldY))), round(surfaceSize[0]/(3*worldX)))
                displayedBackground.blit(gridSquares[worldValues['agent1Pos'][0]][worldValues['agent1Pos'][1]],
                                         squares[worldValues['agent1Pos'][0]][worldValues['agent1Pos'][1]])
                # gridSquares[agentOldPos[0]][agentOldPos[1]].fill(pygame.Color(224,224,255))
                # re-colour visited cells
                gridSquares[agentOldPos[0]][agentOldPos[1]].fill(
                    pygame.Color(210, 150, 150))

                pygame.draw.rect(gridSquares[agentOldPos[0]][agentOldPos[1]], pygame.Color(
                    0, 0, 0), squares[0][0], 5)
                displayedBackground.blit(
                    gridSquares[agentOldPos[0]][agentOldPos[1]], squares[agentOldPos[0]][agentOldPos[1]])
                redraw = True
                agentOldPos[0] = worldValues['agent1Pos'][0]
                agentOldPos[1] = worldValues['agent1Pos'][1]
            for target in worldValues['targets']:
                if target[2]:
                    redraw = True
                    pygame.draw.circle(gridSquares[target[0]][target[1]], pygame.Color(128, 128, 128), (round(
                        surfaceSize[0]/(2*worldX)), round(surfaceSize[1]/(2*worldY))), round(surfaceSize[0]/(4*worldX)))
                    displayedBackground.blit(
                        gridSquares[target[0]][target[1]], squares[target[0]][target[1]])
                elif agentOldPos[0] == target[0] and agentOldPos[1] == target[1]:
                    pygame.draw.circle(gridSquares[target[0]][target[1]], pygame.Color(0, 128, 128), (round(
                        surfaceSize[0]/(2*worldX)), round(surfaceSize[1]/(2*worldY))), round(surfaceSize[0]/(4*worldX)))
                    displayedBackground.blit(
                        gridSquares[target[0]][target[1]], squares[target[0]][target[1]])
            if redraw:
                displaySurface.blit(displayedBackground, backgroundRect)
                pygame.display.flip()
            curTime += 1
