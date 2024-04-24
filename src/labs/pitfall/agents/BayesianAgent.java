package src.labs.pitfall.agents;


// SYSTEM IMPORTS
import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.model.history.History.HistoryView;
import edu.cwru.sepia.environment.model.state.State.StateView;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;


import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;


// JAVA PROJECT IMPORTS
import edu.bu.labs.pitfall.Difficulty;
import edu.bu.labs.pitfall.Synchronizer;
import edu.bu.labs.pitfall.utilities.Coordinate;



public class BayesianAgent
    extends Agent
{

    public static class PitfallBayesianNetwork
        extends Object
    {
        private Map<Coordinate, Boolean>    knownBreezeCoordinates;
        private Set<Coordinate>             frontierPitCoordinates;
        private Set<Coordinate>             otherPitCoordinates;
        private final double                pitProb;

        public PitfallBayesianNetwork(Difficulty difficulty)
        {
            this.knownBreezeCoordinates = new HashMap<Coordinate, Boolean>();

            this.frontierPitCoordinates = new HashSet<Coordinate>();
            this.otherPitCoordinates = new HashSet<Coordinate>();

            this.pitProb = Difficulty.getPitProbability(difficulty);
        }

        public Map<Coordinate, Boolean> getKnownBreezeCoordinates() { return this.knownBreezeCoordinates; }
        public Set<Coordinate> getFrontierPitCoordinates() { return this.frontierPitCoordinates; }
        public Set<Coordinate> getOtherPitCoordinates() { return this.otherPitCoordinates; }
        public final double getPitProb() { return this.pitProb; }


        /**
         *  TODO: please replace this code. The code here will pick a **random** frontier square to explore next,
         *        which may be a pit! You should do the following steps:
         *          1) for each frontier square X, calculate the query Pr[Pit_X = true | evidence]
         *             we typically expand this to say:
         *                         Pr[Pit_X = true | evidence] = alpha * Pr[Pit_X = true && evidence]
         *             however you don't need to calculate alpha explicitly.
         *             If you calculate Pr[Pit_X = true && evidence] for every X, you can convert the values into
         *             probabilities by adding up all Pr[Pit_X = true && evidence] values and dividing each
         *             Pr[Pit_X = true && evidence] value by the sum.
         *
         *          2) pick the pit that is the least likely to have a pit in it to explore next!
         *
         *          As an aside here, you can certainly choose to calculate Pr[Pit_X = false | evidence] values
         *          instead (and then pick the coordinate with the highest prob), its up to you!
         **/
        public Boolean hasNeighboringBreeze(Coordinate c){
            Coordinate north = new Coordinate(c.getXCoordinate(), c.getYCoordinate() + 1);
            Coordinate south = new Coordinate(c.getXCoordinate(), c.getYCoordinate() - 1);
            Coordinate east = new Coordinate(c.getXCoordinate() + 1, c.getYCoordinate());
            Coordinate west = new Coordinate(c.getXCoordinate() - 1, c.getYCoordinate());
            if(this.knownBreezeCoordinates.containsKey(north) && this.knownBreezeCoordinates.get(north)==true || this.knownBreezeCoordinates.containsKey(south) && this.knownBreezeCoordinates.get(south)==true 
            || this.knownBreezeCoordinates.containsKey(east) && this.knownBreezeCoordinates.get(east)==true || this.knownBreezeCoordinates.containsKey(west) && this.knownBreezeCoordinates.get(west)==true){
                return true;
            }
            return false;
        }
        public Set<Coordinate> getFrontierAdjacentCoordinates(Set<Coordinate> frontier){
            Set<Coordinate> adjacents = new HashSet<Coordinate>();
            for(Coordinate c : frontier){
                Coordinate north = new Coordinate(c.getXCoordinate(), c.getYCoordinate() + 1);
                Coordinate south = new Coordinate(c.getXCoordinate(), c.getYCoordinate() - 1);
                Coordinate east = new Coordinate(c.getXCoordinate() + 1, c.getYCoordinate());
                Coordinate west = new Coordinate(c.getXCoordinate() - 1, c.getYCoordinate());
                if(getKnownBreezeCoordinates().containsKey(north)){
                    if(!adjacents.contains(north)){
                        adjacents.add(north);
                    }
                }
                if(getKnownBreezeCoordinates().containsKey(west)){
                    if(!adjacents.contains(west)){
                        adjacents.add(west);
                    }
                }
                if(getKnownBreezeCoordinates().containsKey(east)){
                    if(!adjacents.contains(east)){
                        adjacents.add(east);
                    }
                }
                if(getKnownBreezeCoordinates().containsKey(south)){
                    if(!adjacents.contains(south)){
                        adjacents.add(south);
                    }
                }
            }
            return adjacents;
        }
        public String binaryCounter(int count, int length) {
            StringBuilder binary = new StringBuilder();
            
            while (count > 0) {
                binary.insert(0, count % 2);
                count /= 2;
            }
            
            while (binary.length() < length) {
                binary.insert(0, "0");
            }
            return binary.toString();
        }
        public ArrayList<String> generateValidConfigurations(int frontierSize, Set<Coordinate> frontier, Set<Coordinate> adjacents){
            ArrayList<String> validConfigs = new ArrayList<String>();
            for(int i=1; i<Math.pow(2, frontierSize); i++){
                String binary = binaryCounter(i, frontierSize);
                Boolean isValidCombo = true;
                for(int j=0; j<adjacents.size(); j++){
                    Coordinate cur = (Coordinate) adjacents.toArray()[j];
                    if(binary.charAt(j) == '1'){
                        if(!hasNeighboringBreeze(cur)){
                            isValidCombo = false;
                            break;
                        }
                    }
                }
                if(isValidCombo ){
                    validConfigs.add(binary);
                }
            }
            return validConfigs;
        }
        public double[] getProbsFromConfigs(ArrayList<String> combinations){
            double[] probs = new double[combinations.size()];
            for(int i=0; i<combinations.size(); i++){
                double prob = 1;
                String binary = combinations.get(i);
                for(int j=0; j<binary.length(); j++){
                    if(binary.charAt(j) == '1'){
                        prob *= this.getPitProb();
                    }else{
                        prob *= (1-this.getPitProb());
                    }
                }
                probs[i] = prob;
            }
            return probs;
        }
        
        
        public Coordinate getNextCoordinateToExplore()
        {
            Coordinate toExplore = null;
            // Map<Coordinate, Boolean> breezes = getKnownBreezeCoordinates();
            Map<Coordinate, Double> pitProbabilities = new HashMap<Coordinate, Double>();
            Set<Coordinate> frontier = getFrontierPitCoordinates();
            double prob = this.getPitProb();
            Set<Coordinate> adjacents = getFrontierAdjacentCoordinates(frontier);
            ArrayList<String> validConfigs = generateValidConfigurations(frontier.size(), frontier);
            double[] probs = getProbsFromConfigs(validConfigs);

            double[] frontierProbs = new double[frontier.size()];

            for(int k=0; k<validConfigs.size(); k++){ //for combo in valid combos
                for(int i = 0; i<validConfigs.get(k).length(); i++){ //for coordinate in frontier
                    if(validConfigs.get(k).charAt(i) == '1'){
                        frontierProbs[i] += probs[k];
                        // System.out.println(probs[k]);
                        // System.out.println(frontierProbs[i]);
                    }
                }
            }

            // System.out.println(validConfigs.size());
            // System.out.println(Arrays.toString(validConfigs.toArray()));

            int smallestProbIndex =0;
            double smallestProb = 1.0;
            for(int i=0; i<frontier.size(); i++){
                if(frontierProbs[i] < smallestProb){
                    smallestProb = frontierProbs[i];
                    smallestProbIndex = i;
                }
            }
            toExplore = (Coordinate) frontier.toArray()[smallestProbIndex];
            System.out.println(Arrays.toString(frontierProbs));
            System.out.println(toExplore.toString());
            
            

            // for(int i = 0; i<frontier.size(); i++)
            // {
            //     Coordinate cur = (Coordinate) frontier.toArray()[i];

            //     if(!hasNeighboringBreeze(cur))
            //     {//if there is no breeze in ANY of the DISCOVERED neighboring squares, then the probability of a pit is 0
            //         pitProbabilities.put(cur, 0.0);
            //     }
            //     else if(hasNeighboringBreeze(cur))
            //     {
            //         double configurationCount = 0.0; //used to keep track of valid combinations
            //         double accumulateProb = 0.0; //used to accumulate the probabilities of all valid combinations
            //         for(int k=0; k<Math.pow(2,frontier.size()); k++){
            //             double thisProb = 1; //used to keep track of the probability of the current configuration
            //             String binary = binaryCounter(k, frontier.size());
            //             // System.out.println(binary);
            //             Boolean isValidCombo = true; //boolean for if the current configuration is valid
            //             //iterate through the binary representation of the configuration
            //             for(int j=0; j<frontier.size(); j++){
            //                 Coordinate front = (Coordinate) frontier.toArray()[j];
            //                 if(binary.charAt(j) == '1'){
            //                     //if the coordinate has adjacent breezes and is set to 1, we consider that it has a pit here
            //                     if(hasNeighboringBreeze(front)){
            //                         thisProb *= prob;
            //                         // configurationCount++;
            //                     }
            //                     //if the coordinate has no breezes but is set to 1, we dont consider it the entire binary string; impossible combination
            //                     else{
            //                         isValidCombo = false;
            //                         continue;
            //                     }
            //                 }
            //                 else{
            //                     //if 'cur' coordinate is set to 0 in this configuration, we dont consider the entire binary string
            //                     if(front.equals(cur)){
            //                         isValidCombo = false;
            //                         continue;
            //                     }
            //                     //if the coord has adjacent breezes and is set to 0, we consider that it doesnt have a pit here
            //                     else if(hasNeighboringBreeze(front)){
            //                         thisProb *= (1-prob);
            //                         // configurationCount++;
            //                     }
            //                     //if the coord has no adjacent breezes and is set to 0, we dont consider the entire binary string; irrelevant combination
            //                     else{
            //                         isValidCombo = false;
            //                         continue;
            //                     }
            //                 }
            //                 if(isValidCombo){
            //                     configurationCount++;
            //                     accumulateProb += thisProb;
            //                 }else{
            //                     isValidCombo = true;
            //                 }
            //             }
            //         }
            //         pitProbabilities.put(cur, accumulateProb/configurationCount);
            //     }
            //     // else
            //     // {
            //     //     pitProbabilities.put(cur, this.getPitProb());
            //     // }
            // }

            // //this part finds the smalled probability coordinate and selects it to be returned
            // toExplore = (Coordinate) pitProbabilities.keySet().toArray()[0];
            // for(int j = 0; j < pitProbabilities.size(); j++){
            //     Coordinate cur = (Coordinate) pitProbabilities.keySet().toArray()[j];
            //     if(pitProbabilities.get(cur) < pitProbabilities.get(toExplore)){
            //         toExplore = cur;
            //     }
            // }
            // System.out.println(breezes.toString());
            // System.out.println(frontier.toString());
            // System.out.println(pitProbabilities.toString());
            // System.out.println(toExplore.toString());
            return toExplore;
        }
    }

    private int                     myUnitID;
    private int                     enemyPlayerNumber;
    private Set<Coordinate>         gameCoordinates;
    private Set<Coordinate>         unexploredCoordinates;
    private Coordinate              coordinateIJustAttacked;
    private Coordinate              srcCoordinate;
    private Coordinate              dstCoordinate;
    private PitfallBayesianNetwork  bayesianNetwork;

    private final Difficulty        difficulty;

	public BayesianAgent(int playerNum, String[] args)
	{
        super(playerNum);

        if(args.length != 3)
		{
			System.err.println("[ERROR] BayesianAgent.BayesianAgent: need to provide args <playerID> <seed> <difficulty>");
		}

        this.myUnitID = -1;
        this.enemyPlayerNumber = -1;
        this.gameCoordinates = new HashSet<Coordinate>();
        this.unexploredCoordinates = new HashSet<Coordinate>();
        this.coordinateIJustAttacked = null;
        this.srcCoordinate = null;
        this.dstCoordinate = null;
        this.bayesianNetwork = null;

        this.difficulty = Difficulty.valueOf(args[2].toUpperCase());
	}

	public int getMyUnitID() { return this.myUnitID; }
    public int getEnemyPlayerNumber() { return this.enemyPlayerNumber; }
    public Set<Coordinate> getGameCoordinates() { return this.gameCoordinates; }
    public Set<Coordinate> getUnexploredCoordinates() { return this.unexploredCoordinates; }
    public final Coordinate getCoordinateIJustAttacked() { return this.coordinateIJustAttacked; }
    public final Coordinate getSrcCoordinate() { return this.srcCoordinate; }
    public final Coordinate getDstCoordinate() { return this.dstCoordinate; }
    public PitfallBayesianNetwork getBayesianNetwork() { return this.bayesianNetwork; }
    public final Difficulty getDifficulty() { return this.difficulty; }

    private void setMyUnitID(int i) { this.myUnitID = i; }
    private void setEnemyPlayerNumber(int i) { this.enemyPlayerNumber = i; }
    private void setCoordinateIJustAttacked(Coordinate c) { this.coordinateIJustAttacked = c; }
    private void setSrcCoordinate(Coordinate c) { this.srcCoordinate = c; }
    private void setDstCoordinate(Coordinate c) { this.dstCoordinate = c; }
    private void setBayesianNetwork(PitfallBayesianNetwork n) { this.bayesianNetwork = n; }

	@Override
	public Map<Integer, Action> initialStep(StateView state,
                                            HistoryView history)
	{

		// locate enemy and friendly units
        Set<Integer> myUnitIDs = new HashSet<Integer>();
		for(Integer unitID : state.getUnitIds(this.getPlayerNumber()))
        {
            myUnitIDs.add(unitID);
        }

        if(myUnitIDs.size() != 1)
        {
            System.err.println("[ERROR] PitfallAgent.initialStep: should only have 1 unit but found "
                + myUnitIDs.size());
            System.exit(-1);
        }

		// check that all units are archers units
	    if(!state.getUnit(myUnitIDs.iterator().next()).getTemplateView().getName().toLowerCase().equals("archer"))
	    {
		    System.err.println("[ERROR] PitfallAgent.initialStep: should only control archers!");
		    System.exit(1);
	    }

        // get the other player
		Integer[] playerNumbers = state.getPlayerNumbers();
		if(playerNumbers.length != 2)
		{
			System.err.println("ERROR: Should only be two players in the game");
			System.exit(-1);
		}
		Integer enemyPlayerNumber = null;
		if(playerNumbers[0] != this.getPlayerNumber())
		{
			enemyPlayerNumber = playerNumbers[0];
		} else
		{
			enemyPlayerNumber = playerNumbers[1];
		}

        // check enemy units
        Set<Integer> enemyUnitIDs = new HashSet<Integer>();
        for(Integer unitID : state.getUnitIds(enemyPlayerNumber))
        {
            if(!state.getUnit(unitID).getTemplateView().getName().toLowerCase().equals("hiddensquare"))
		    {
			    System.err.println("ERROR [BayesianAgent.initialStep]: Enemy should start off with HiddenSquare units!");
			        System.exit(-1);
		    }
            enemyUnitIDs.add(unitID);
        }


        // initially everything is unknown
        Coordinate coord = null;
        for(Integer unitID : enemyUnitIDs)
        {
            coord = new Coordinate(state.getUnit(unitID).getXPosition(),
                                   state.getUnit(unitID).getYPosition());
            this.getUnexploredCoordinates().add(coord);
            this.getGameCoordinates().add(coord);
        }

        this.setMyUnitID(myUnitIDs.iterator().next());
        this.setEnemyPlayerNumber(enemyPlayerNumber);
        this.setSrcCoordinate(new Coordinate(1, state.getYExtent() - 2));
        this.setDstCoordinate(new Coordinate(state.getXExtent() - 2, 1));
        this.setBayesianNetwork(new PitfallBayesianNetwork(this.getDifficulty()));

        Map<Integer, Action> initialActions = new HashMap<Integer, Action>();
        initialActions.put(
            this.getMyUnitID(),
            Action.createPrimitiveAttack(
                this.getMyUnitID(),
                state.unitAt(this.getSrcCoordinate().getXCoordinate(), this.getSrcCoordinate().getYCoordinate())
            )
        );
        this.getUnexploredCoordinates().remove(this.getSrcCoordinate());
		return initialActions;
	}

    public boolean isFrontierCoordiante(Coordinate src,
                                        StateView state)
    {
        int dirs[][] = new int[][]{{-1, 0}, {+1, 0}, {0, -1}, {0, +1}};
        for(int dir[] : dirs)
        {
            int x = src.getXCoordinate() + dir[0];
            int y = src.getYCoordinate() + dir[1];

            if(x >= 1 && x <= state.getXExtent() - 2 &&
               y >= 1 && y <= state.getYExtent() - 2 &&
               (!state.isUnitAt(x, y) ||
                !state.getUnit(state.unitAt(x, y)).getTemplateView().getName().toLowerCase().equals("hiddensquare")))
            {
                return true;
            }
        }
        return false;
    }

    public void makeObservations(StateView state,
                                 HistoryView history)
    {
        this.getBayesianNetwork().getKnownBreezeCoordinates().clear();
        this.getBayesianNetwork().getFrontierPitCoordinates().clear();
        this.getBayesianNetwork().getOtherPitCoordinates().clear();

        Set<Coordinate> exploredCoordinates = new HashSet<Coordinate>();
        for(Integer enemyUnitID : state.getUnitIds(this.getEnemyPlayerNumber()))
        {
            UnitView enemyUnitView = state.getUnit(enemyUnitID);
            if(enemyUnitView.getTemplateView().getName().toLowerCase().equals("breezesquare"))
            {
                this.getBayesianNetwork().getKnownBreezeCoordinates().put(
                    new Coordinate(enemyUnitView.getXPosition(),
                                   enemyUnitView.getYPosition()),
                    true
                );
            } else if(enemyUnitView.getTemplateView().getName().toLowerCase().equals("safesquare"))
            {
                this.getBayesianNetwork().getKnownBreezeCoordinates().put(
                    new Coordinate(enemyUnitView.getXPosition(),
                                   enemyUnitView.getYPosition()),
                    false
                );
            } else if(enemyUnitView.getTemplateView().getName().toLowerCase().equals("hiddensquare"))
            {
                this.getBayesianNetwork().getOtherPitCoordinates().add(
                    new Coordinate(enemyUnitView.getXPosition(),
                                   enemyUnitView.getYPosition())
                );
            }

            // now separate out the frontier from the "other" ones
            for(Coordinate unknownCoordinate : this.getBayesianNetwork().getOtherPitCoordinates())
            {
                if(this.isFrontierCoordiante(unknownCoordinate, state))
                {
                    this.getBayesianNetwork().getFrontierPitCoordinates().add(unknownCoordinate);
                }
            }
            this.getBayesianNetwork().getOtherPitCoordinates().removeAll(
                this.getBayesianNetwork().getFrontierPitCoordinates()
            );
        }
    }

	@Override
	public Map<Integer, Action> middleStep(StateView state,
                                           HistoryView history) {
		Map<Integer, Action> actions = new HashMap<Integer, Action>();

        if(Synchronizer.isMyTurn(this.getPlayerNumber(), state))
        {

            // get the observation from the past
            if(state.getTurnNumber() > 0)
            {
                this.makeObservations(state, history);
            }

            Coordinate coordinateOfUnitToAttack = this.getBayesianNetwork().getNextCoordinateToExplore();

            // could have won the game (and waiting for enemy units to die)
            // or we have a coordinate to attack
            // we need to check that the unit at that coordinate is a hidden square (not allowed to attack other units)
            if(coordinateOfUnitToAttack != null)
            {
                Integer unitID = state.unitAt(coordinateOfUnitToAttack.getXCoordinate(),
                                              coordinateOfUnitToAttack.getYCoordinate());
                if(unitID == null)
                {
                    System.err.println("ERROR: BayesianAgent.middleStep: deciding to attack unit at " +
                        coordinateOfUnitToAttack + " but no unit was found there!");
                    System.exit(-1);
                }

                String unitTemplateName = state.getUnit(unitID).getTemplateView().getName();
                if(!unitTemplateName.toLowerCase().equals("hiddensquare"))
                {
                    // can't attack non hidden-squares!
                    System.err.println("ERROR: BayesianAgent.middleStep: deciding to attack unit at " +
                        coordinateOfUnitToAttack + " but unit at that square is [" + unitTemplateName + "] " +
                        "and should be a HiddenSquare unit!");
                    System.exit(-1);
                }
                this.setCoordinateIJustAttacked(coordinateOfUnitToAttack);

                actions.put(
                    this.getMyUnitID(),
                    Action.createPrimitiveAttack(
                        this.getMyUnitID(),
                        unitID)
                );
                this.getUnexploredCoordinates().remove(coordinateOfUnitToAttack);
            }

        }

		return actions;
	}

    @Override
	public void terminalStep(StateView state, HistoryView history) {}

    @Override
	public void loadPlayerData(InputStream arg0) {}

	@Override
	public void savePlayerData(OutputStream arg0) {}

}

