package src.pas.tetris.agents;


// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Arrays;
import java.util.ArrayList;


// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.Module;
import edu.bu.tetris.utils.Coordinate;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;


public class TetrisQAgent
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.05;
    public static int expeditionCount = 1;

    private Random random;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction() {
        final int inputSize = 5; 
        final int hiddenDim = (int)Math.pow(inputSize, 2); //danger variable >:)
        final int outDim = 1; 
        Sequential qFunction = new Sequential();
        // qFunction.add(new ReLU());
        qFunction.add(new Dense(inputSize, hiddenDim));
        qFunction.add(new ReLU()); //relu supremacy
        qFunction.add(new Dense(hiddenDim, outDim));
        return qFunction;
    }

    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */
    @Override
    public Matrix getQFunctionInput(final GameView game, final Mino potentialAction) {
        Matrix gameMatrix = null; 
        Matrix featureMatrix = Matrix.zeros(1, 5);

        
        int bumpiness = 0; // sum of the absolute differences in height between adjacent columns
        Integer[] colHeights = new Integer[10];
        int fullRows = 0; //number of complete rows
        int emptyBelow = 0;
        
        int baseHeight = 22; //height of the tallest column
        boolean isBaseHeightSet = false;
        int minoShape = -1;
        Mino.MinoType curMino = potentialAction.getType(); 

        try{
            gameMatrix = game.getGrayscaleImage(potentialAction); //getgrayscaleimage my beloved
        }catch(Exception oops){
            System.out.println("NOOOOO!!!! THIS CANNOT BE!!!!!!");
            oops.printStackTrace();
            System.exit(-1);
        }
        
        for(int x = 0; x < gameMatrix.getShape().getNumRows(); x++ ){//grayscale matrix my love my only
            boolean isFull = true;
            for(int y = 0; y < gameMatrix.getShape().getNumCols(); y++){
                if(gameMatrix.get(x, y) == 0.5 || gameMatrix.get(x, y) == 1.0){ //if the square is occupied
                    if (isBaseHeightSet == false ){
                        baseHeight = x;
                        isBaseHeightSet = true;
                    }
                    if(colHeights[y] == null){ 
                        colHeights[y] = x;
                    }
                    if(y == 9 && isFull == true){
                        fullRows += 1;
                    }
                }
                else if(gameMatrix.get(x, y) == 0.0){//if the square is empty
                    if(colHeights[y] != null){
                        emptyBelow += 1;
                    }
                    isFull = false;
                }
            }
        }

        for (int i = 0; i < 9; i++) {
            int cur = 22;
            if (colHeights[i] != null) {
                cur = colHeights[i];
            }

            int next = 22;
            if (colHeights[i+1] != null) {
                next = colHeights[i+1];
            }
            bumpiness += Math.abs(cur - next);
        }

        if(curMino == Mino.MinoType.valueOf("I")){
            minoShape = 0;
        }
        else if(curMino == Mino.MinoType.valueOf("J")){
            minoShape = 1;
        }
        else if(curMino == Mino.MinoType.valueOf("L")){
            minoShape = 2;
        }
        else if(curMino == Mino.MinoType.valueOf("O")){
            minoShape = 3;
        }
        else if(curMino == Mino.MinoType.valueOf("S")){
            minoShape = 4;
        }
        else if(curMino == Mino.MinoType.valueOf("T")){
            minoShape = 5;
        }
        else if(curMino == Mino.MinoType.valueOf("Z")){
            minoShape = 6;
        }
        featureMatrix.set(0, 0, baseHeight);
        featureMatrix.set(0, 1, bumpiness);
        featureMatrix.set(0, 2, emptyBelow);
        featureMatrix.set(0, 3, fullRows);
        featureMatrix.set(0, 4, minoShape);

        return featureMatrix;
    }

    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game, final GameCounter gameCounter) {


        int turnIdx = (int)gameCounter.getCurrentMoveIdx();// which turn # this is in a given game
        int gameIdx = (int)gameCounter.getCurrentGameIdx();//which game # this is

        double explorationFloor = 0.05;
        double initExploreProb = 1.0;
        double exploreDecayRate = 0.99 - (gameIdx * 0.001);

        double currentExplorationProb = Math.max(explorationFloor, initExploreProb - turnIdx * (initExploreProb - explorationFloor) / exploreDecayRate);
        return this.getRandom().nextDouble() <= currentExplorationProb;
    }

    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game) {
        int numPossibilities = game.getFinalMinoPositions().size();
        Matrix outcomes = Matrix.zeros(1, numPossibilities);

        for(int i = 0; i < numPossibilities; i++){
            try{
                Matrix cur = this.getQFunctionInput(game, game.getFinalMinoPositions().get(i));
                outcomes.set(0, i, Math.exp(this.initQFunction().forward(cur).get(0, 0)));
            } catch (Exception oops) {
                oops.printStackTrace();
                System.exit(-1);
            }
        }
        int minValPos = 0;
        double minVal = Double.POSITIVE_INFINITY;
        double qSum = outcomes.sum().get(0, 0);
        Matrix qMatrix = Matrix.zeros(1, 1);
        Matrix outcome = null;
        
        qMatrix.set(0, 0, qSum);
        try {
            outcome = outcomes.ediv(qMatrix);
        } catch (Exception oops) {
            oops.printStackTrace();
            System.exit(-1);
        }
        for (int i = 0; i < numPossibilities; i++) {
            if (outcome.get(0, i) < minVal) {
                minVal = outcome.get(0, i);
                minValPos = i;
            }
        }
        return game.getFinalMinoPositions().get(minValPos);
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset, LossFunction lossFunction, Optimizer optimizer, long numUpdates) {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx){
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
    public double getReward(final GameView game) {
        Board b = game.getBoard();
        double reward = 0.0;

        int largestYVal = 22;
        Coordinate highestCord = null;
        Boolean isHighestCoordSet = false;
        
        int numSpacesBelowHighestMino = 0;
        Integer[] colMax = new Integer[10];

        int sandpaper = 0; // sum of the absolute differences in height between adjacent columns

        int completedRows = 0;

        // parse through board to collect feature data
        for (int y = 0; y < 22; y++) {
            boolean isFullRow = true;
            for (int x = 0; x < 10; x++) {
                // occupied coordinate
                if (b.isCoordinateOccupied(x, y)) { 
                    // get the highest coordinate position
                    if (isHighestCoordSet == false) {
                        highestCord = new Coordinate(x, y);
                        isHighestCoordSet = true;
                    }
                    // sets the max for the current column
                    if (colMax[x] == null) {
                        colMax[x] = y;
                    }
                    isFullRow = false;
                }
                // empty coordinate
                else if (!b.isCoordinateOccupied(x, y)) {
                    if(colMax[x] != null) {
                        numSpacesBelowHighestMino += 1;
                    }
                    if(x == 9 && isFullRow && isHighestCoordSet){
                        completedRows += 1;
                    }
                }
            }
            if (highestCord != null) {
                isHighestCoordSet = true;
            }
        }

        // set highestY
        if (highestCord != null) {
            largestYVal = highestCord.getYCoordinate();
        }

        // set sandpaper
        for (int i = 0; i < 9; i++) {
            int cur = 22;
            if (colMax[i] != null) {
                cur = colMax[i];
            }

            int next = 22;
            if (colMax[i+1] != null) {
                next = colMax[i+1];
            }
            sandpaper += Math.abs(cur - next);
        }

        // set ptsEarned
        int ptsEarned = game.getScoreThisTurn();


        // prints for each feature data point
        //System.out.println("ptsEarned: " + ptsEarned);
        //System.out.println("largestYVal: " + largestYVal);
        //System.out.println("filledDensity: " + filledDensity);
        //System.out.println("numSpacesBelowHighestMino: " + numSpacesBelowHighestMino);
        //System.out.println("heightDelta: " + heightDelta);
        //System.out.println("spaceWithMinoBeneath: " + spaceWithMinoBeneath);
        //System.out.println("sandpaper: " + sandpaper);


        // reward = Math.pow((5 * ptsEarned),1.5) - ((spaceWithMinoBeneath * .5) + (lowestEmptyYPos * .2) + (numSpacesBelowHighestMino * .2) + (sandpaper * .1));
        // reward = (10 * ptsEarned + largestYVal) - (Math.pow(1-filledDensity, 4) * ((numSpacesBelowHighestMino * 7) + (sandpaper * 3)));
        reward = ((50 * ptsEarned) + (10 * completedRows)) - ((sandpaper + numSpacesBelowHighestMino) / (double)largestYVal);


        return reward;
    }
}