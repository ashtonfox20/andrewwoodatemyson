package src.pas.battleship.agents;


// SYSTEM IMPORTS


// JAVA PROJECT IMPORTS
import edu.bu.battleship.agents.Agent;
import edu.bu.battleship.game.Game.GameView;
import edu.bu.battleship.game.ships.Ship;
import edu.bu.battleship.game.EnemyBoard;
import edu.bu.battleship.game.Game;
import edu.bu.battleship.game.EnemyBoard.Outcome;
import edu.bu.battleship.utils.Coordinate;

import java.util.ArrayList;
import java.util.Map;
import java.util.Random;


public class ProbabilisticAgent
    extends Agent
{
    static ArrayList<Coordinate> pastGuesses = new ArrayList<Coordinate>();
    static Boolean hunting = false; //turns on when a ship is hit, turns off when a ship is sunk. used for incinerate protocol
    static Coordinate hitCoord; //used to store the coordinate of the first hit on a ship - resets when a shit gets sunk
    static int turnCount = 0;
    static ArrayList<Coordinate> potentialHitGuesses = new ArrayList<Coordinate>(); //when trying to sink a found ship, this gets used




    public ProbabilisticAgent(String name)
    {
        super(name);
        System.out.println("[INFO] ProbabilisticAgent.ProbabilisticAgent: constructed agent");
    }
    public Boolean isIn(ArrayList<Coordinate> list, Coordinate c){
        if(list.size()==0){
            return false;
        }
        for(Coordinate coord : list){
            if(coord.equals(c)){
                return true;
            }
        }
        return false;
    }
    public ArrayList<Coordinate> getAllPotentialSpots(Coordinate c, GameView game, ArrayList<Coordinate> pastGuesses, int maxShipSize){
        //get all potential spots around a hit coordinate, based on the maximum ship size still alive on the board
        //only called to generate all potential spots a ship could be in based on the FIRST fresh hit
        ArrayList<Coordinate> potentialSpots = new ArrayList<Coordinate>();
        for(int i = 1; i < maxShipSize+1; i++){//loop for northside coords
            Coordinate north = new Coordinate(c.getXCoordinate(), c.getYCoordinate() + i);
            if(game.isInBounds(north.getXCoordinate(),north.getYCoordinate()) && !isIn(pastGuesses, north)){
                potentialSpots.add(north);
            }else{
                break;
        }}
        for(int i = 1; i < maxShipSize+1; i++){//loop for southside coords
            Coordinate south = new Coordinate(c.getXCoordinate(), c.getYCoordinate() - i);
            if(game.isInBounds(south.getXCoordinate(),south.getYCoordinate()) && !isIn(pastGuesses, south)){
                potentialSpots.add(south);
            }else{
                break;
        }}
        for(int i = 1; i < maxShipSize+1; i++){//loop for eastside coords
            Coordinate east = new Coordinate(c.getXCoordinate() + i, c.getYCoordinate());
            if(game.isInBounds(east.getXCoordinate(),east.getYCoordinate()) && !isIn(pastGuesses, east)){
                potentialSpots.add(east);
            }else{
                break;
        }}
        for(int i = 1; i < maxShipSize+1; i++){//loop for westside coords
            Coordinate west = new Coordinate(c.getXCoordinate() - i, c.getYCoordinate());
            if(game.isInBounds(west.getXCoordinate(),west.getYCoordinate()) && !isIn(pastGuesses, west)){
                potentialSpots.add(west);
            }else{
                break;
        }}
        return potentialSpots;
    }

    public int getMaxSizeOfShipAlive(Map<Ship.ShipType, Integer> ships){
        if(ships.get(Ship.ShipType.PATROL_BOAT) > 0){
            return 2;
        }else if(ships.get(Ship.ShipType.PATROL_BOAT) == 0 && ships.get(Ship.ShipType.DESTROYER) > 0 || ships.get(Ship.ShipType.SUBMARINE) > 0){
            return 3;
        }else if(ships.get(Ship.ShipType.PATROL_BOAT) == 0 && ships.get(Ship.ShipType.DESTROYER) == 0 && ships.get(Ship.ShipType.SUBMARINE) == 0 && ships.get(Ship.ShipType.BATTLESHIP) > 0){
            return 4;
        }else if(ships.get(Ship.ShipType.PATROL_BOAT) == 0 && ships.get(Ship.ShipType.DESTROYER) == 0 && ships.get(Ship.ShipType.SUBMARINE) == 0 && ships.get(Ship.ShipType.BATTLESHIP) == 0 && ships.get(Ship.ShipType.AIRCRAFT_CARRIER) > 0){
            return 5;
        }else{
            return 2;
        }
    }
    public int getBiggestShipSize(Map<Ship.ShipType, Integer> ships){
        if(ships.get(Ship.ShipType.AIRCRAFT_CARRIER) > 0){
            return 5;
        }else if(ships.get(Ship.ShipType.BATTLESHIP) > 0){
            return 4;
        }else if(ships.get(Ship.ShipType.SUBMARINE) > 0 || ships.get(Ship.ShipType.DESTROYER) > 0){
            return 3;
        }else if(ships.get(Ship.ShipType.PATROL_BOAT) > 0){
            return 2;
        }else{
            return 2;
        }
    }
    public ArrayList<Coordinate> populateMatrix(int ylimit, int xlimit, int targetShipSize){
        ArrayList<Coordinate> potentialGuesses = new ArrayList<Coordinate>();
        int x;
        int y = 0;
        // System.out.println(targetShipSize);
        while(y < ylimit){
            for(x = 0; x < xlimit; x++){
                if((x+y)%targetShipSize == 0){
                    
                    // System.out.println("Iteration " + x + " adding coordinate " +x%xlimit+","+y+ " y = " + y);
                    Coordinate c = new Coordinate(x, y);
                    potentialGuesses.add(c);
                }
            }
            y++;
        }
        // System.out.println("Made it out of populateMatrix");
        return potentialGuesses;
    }
    public ArrayList<Coordinate> populateMatrixAlt(int ylimit, int xlimit, int targetShipSize){
        ArrayList<Coordinate> potentialGuesses = new ArrayList<Coordinate>();
        int x;
        int y = 0;
        // System.out.println(targetShipSize);
        while(y < ylimit){
            for(x = 0; x < xlimit; x++){
                if((x+y)%targetShipSize != 0){
                    
                    // System.out.println("Iteration " + x + " adding coordinate " +x%xlimit+","+y+ " y = " + y);
                    Coordinate c = new Coordinate(x, y);
                    potentialGuesses.add(c);
                }
            }
            y++;
        }
        // System.out.println("Made it out of populateMatrix");
        return potentialGuesses;
    }
    public ArrayList<Coordinate> isViableOption(ArrayList<Coordinate> potentialGuesses, int targetShipSize, GameView game, ArrayList<Coordinate> pastGuesses){
        //if a ship of the target size can fit in any configuration including the guess coordinate, it is viable; parse if not viable
        ArrayList<Coordinate> viableOptions = new ArrayList<Coordinate>();
        for(Coordinate c : potentialGuesses){
            int x = c.getXCoordinate();
            int y = c.getYCoordinate();
            Coordinate cCoord = new Coordinate(x,y);
            int west = 0;
            int east = 0;
            int north = 0;
            int south = 0;
            
            if(isIn(pastGuesses, cCoord)){
                continue;
            }
            for(int i = 1; i < targetShipSize; i++){
                Coordinate westCoord = new Coordinate(x-i, y);
                if((game.isInBounds(westCoord.getXCoordinate(),westCoord.getYCoordinate())&& !isIn(pastGuesses, westCoord))){
                    west++;
                }}
            for(int i = 1; i < targetShipSize; i++){
                Coordinate eastCoord = new Coordinate(x+i, y);
                if((game.isInBounds(eastCoord.getXCoordinate(),eastCoord.getYCoordinate())&& !isIn(pastGuesses, eastCoord))){
                    east++;
                }}
            for(int i = 1; i < targetShipSize; i++){
                Coordinate northCoord = new Coordinate(x, y+i);
                if((game.isInBounds(northCoord.getXCoordinate(),northCoord.getYCoordinate())&& !isIn(pastGuesses, northCoord))){
                    north++;
                }}
            for(int i = 1; i < targetShipSize; i++){
                Coordinate southCoord = new Coordinate(x, y-i);
                if((game.isInBounds(southCoord.getXCoordinate(),southCoord.getYCoordinate())&& !isIn(pastGuesses, southCoord))){
                    south++;
                }}
            // System.out.println(west + " " + east + " " + north + " " + south);
            if((west+east+1) >= targetShipSize || (north+south+1) >= targetShipSize){
                viableOptions.add(c);
            }
        }
        // System.out.println("Made it out of isViableOption");
        return viableOptions;
    }

    @Override
    public Coordinate makeMove(final GameView game)
    {
        // System.out.println("[INFO] ProbabilisticAgent.makeMove: making move");
        Map<Ship.ShipType, Integer> ships = game.getEnemyShipTypeToNumRemaining();
        int maxShipSize = getMaxSizeOfShipAlive(ships);
        int biggestShip = getBiggestShipSize(ships);
        EnemyBoard.Outcome[][] prevOutcomes = game.getEnemyBoardView();
        Random rand = new Random(); //will be used to randomly select a coordinate from the potential guesses
        ArrayList<Coordinate> potentialGuesses = new ArrayList<Coordinate>(); //used to create matrices of valid guessing locations based on smallest ship still alive
        int xLimit = game.getGameConstants().getNumCols();
        int yLimit = game.getGameConstants().getNumRows();//variables for the size of the playing grid
        Coordinate guess; //the returned coordinate
        Coordinate lastGuess = new Coordinate(0,0); //initialize the last guess to 0,0
        if(pastGuesses.size() > 0){
            lastGuess = pastGuesses.get(pastGuesses.size()-1); //the last guess made
        }

    
        if(turnCount != 0 && !hunting){
            if(prevOutcomes[lastGuess.getXCoordinate()][lastGuess.getYCoordinate()] == Outcome.HIT){
                hunting = true;
                hitCoord = (pastGuesses.get(turnCount-1));
                potentialHitGuesses = getAllPotentialSpots(pastGuesses.get(turnCount-1), game, pastGuesses, biggestShip);
                System.out.println("[Player1] Protocol: Incinerate");
            }}
        //protocol for hunting down a hit ship
        while(hunting){
            if(prevOutcomes[lastGuess.getXCoordinate()][lastGuess.getYCoordinate()] == Outcome.SUNK){
            hunting = false;
            hitCoord = null; //reset the known hit coordinates for next victim ship
            // potentialHitGuesses = new ArrayList<Coordinate>(); //same with potential guesses
            System.out.println("[Player1] Incineration complete. Reanalyzing... ");
            break;
            }
            if(prevOutcomes[lastGuess.getXCoordinate()][lastGuess.getYCoordinate()] == Outcome.MISS){//if last guess was a miss, prune the list of potential guesses
                System.out.println("[Player1] Reassessing information...");
                int firstHitX = hitCoord.getXCoordinate();
                int firstHitY = hitCoord.getYCoordinate();
                int missHitX = pastGuesses.get(turnCount-1).getXCoordinate();
                int missHitY = pastGuesses.get(turnCount-1).getYCoordinate();
                // System.out.println("hitCoord: " + hitCoord.getXCoordinate() + "," + hitCoord.getYCoordinate());
                // System.out.println("missed hit: " + pastGuesses.get(turnCount-1).getXCoordinate() + "," + pastGuesses.get(turnCount-1).getYCoordinate());
                // System.out.println(potentialHitGuesses.size());
                // for(Coordinate coord: potentialHitGuesses){
                //     System.out.println(coord.getXCoordinate() + "," + coord.getYCoordinate());
                // }
                if(potentialHitGuesses.size() == 0){
                    hunting = false;
                    hitCoord = null;
                    System.out.println("[Player1] Error: Enemy Evaded. Reanalyzing...");
                    break;
                }
                //if the miss was to the left, remove all potential guesses to the left of the first hit
                if(missHitX < firstHitX){
                    for(int i = potentialHitGuesses.size()-1; i == 0; i--){
                        if(potentialHitGuesses.get(i).getXCoordinate() < firstHitX){
                            potentialHitGuesses.remove(i);
                            i++;
                            }}
                }//if the miss hit was to the right, remove all potential guesses to the right of the first hit
                else if(missHitX > firstHitX){
                    for(int i = potentialHitGuesses.size()-1; i == 0; i--){
                        if(potentialHitGuesses.get(i).getXCoordinate() > firstHitX){
                            potentialHitGuesses.remove(i);
                            i++;
                            }}
                }//if the miss hit was above the first hit, remove all potential guesses above the first hit
                else if(missHitY > firstHitY){
                    for(int i = potentialHitGuesses.size()-1; i == 0; i--){
                        if(potentialHitGuesses.get(i).getYCoordinate() > firstHitY){
                            potentialHitGuesses.remove(i);
                            i++;
                        }}
                }//if the miss hit was below the first hit, remove all potential guesses below the first hit
                else if(missHitY < firstHitY){
                    for(int i = potentialHitGuesses.size()-1; i == 0; i--){
                        if(potentialHitGuesses.get(i).getYCoordinate() < firstHitY){
                            potentialHitGuesses.remove(i);
                            i++;
                            }}
                }
                // System.out.println("after parsing");
                // for(Coordinate coord: potentialHitGuesses){
                //     System.out.println(coord.getXCoordinate() + "," + coord.getYCoordinate());
                // }
                // System.out.println("size of potentialHitGuesses: " + potentialHitGuesses.size());
                if(potentialHitGuesses.size() == 0){
                    hunting = false;
                    hitCoord = null;
                    System.out.println("[Player1] Error: Enemy Evaded. Reanalyzing...");
                    break;
                }
                guess = potentialHitGuesses.get(0);
                potentialHitGuesses.remove(guess);
                pastGuesses.add(guess);
                turnCount++;
                return guess;                
            }
            //if the last guess was a hit and wasnt the very first hit, prune the potential guesses to only have guesses along the same cardinal direction
            else if(prevOutcomes[lastGuess.getXCoordinate()][lastGuess.getYCoordinate()] == Outcome.HIT && !pastGuesses.get(turnCount-1).equals(hitCoord)){
                System.out.println("[Player1] Narrowing possibilities...");
                int firstHitX = hitCoord.getXCoordinate();
                int firstHitY = hitCoord.getYCoordinate();
                int hitX = pastGuesses.get(turnCount-1).getXCoordinate();
                int hitY = pastGuesses.get(turnCount-1).getYCoordinate();
                if(potentialHitGuesses.size() == 0){
                    hunting = false;
                    hitCoord = null;
                    System.out.println("[Player1] Error: Enemy Evaded. Reanalyzing...");
                    break;
                }
                //if the hit was on the x-axis, remove all potential guesses that are not on the x-axis
                if(hitX == firstHitX){
                    for(int i = potentialHitGuesses.size()-1; i ==0; i--){
                        if(potentialHitGuesses.get(i).getXCoordinate() != firstHitX){
                            potentialHitGuesses.remove(i);
                            }}
                }//if the hit was on the y-axis, remove all potential guesses that are not on the y-axis
                if(hitY == firstHitY){
                    for(int i = potentialHitGuesses.size()-1; i ==0; i--){
                        if(potentialHitGuesses.get(i).getYCoordinate() != firstHitY){
                            potentialHitGuesses.remove(i);
                            }}
                }
                // for(int i = 0; i < potentialHitGuesses.size(); i++){
                //     if(potentialHitGuesses.get(i).equals(hitCoord)){
                //         potentialHitGuesses.remove(i);
                //     }
                // }
                // System.out.println("size of potentialHitGuesses: " + potentialHitGuesses.size());
                // System.out.println("potentialHitGuess(0) = " + potentialHitGuesses.get(0).getXCoordinate() + "," + potentialHitGuesses.get(0).getYCoordinate());
                if(potentialHitGuesses.size() == 0){
                    hunting = false;
                    hitCoord = null;
                    System.out.println("[Player1] Error: Enemy Evaded. Reanalyzing...");
                    break;
                }
                guess = potentialHitGuesses.get(0);
                // System.out.println("Guess: " + guess.getXCoordinate() + "," + guess.getYCoordinate());
                potentialHitGuesses.remove(guess);
                pastGuesses.add(guess);
                turnCount++;
                return guess; 
            }
            else{//at this point, the only other scenario is if the last guess was the first hit on a ship, just pick the first thing in potential guesses
                if(potentialHitGuesses.size() == 0){
                    hunting = false;
                    hitCoord = null;
                    System.out.println("[Player1] Protocol: Error: Enemy Evaded. Reanalyzing...");
                    break;
                }
                guess = potentialHitGuesses.get(0);
                potentialHitGuesses.remove(guess);
                pastGuesses.add(guess);
                turnCount++;
                return guess; 
            }
        }
        //protocol for seeking patrol ships on the board
        // int protocol = rand.nextInt(100);
        if (maxShipSize == 2)
        {
            System.out.println("[Player1] Protocol: Crush Them");
        }
        else if(maxShipSize == 3){
            System.out.println("[Player1] Protocol: Minefield");
        }
        else if(maxShipSize == 4){
            System.out.println("[Player1] Protocol: Scatter");
        }
        else if(maxShipSize == 5){
            System.out.println("[Player1] Protocol: Rip & Tear");
        }
        potentialGuesses = populateMatrix(yLimit, xLimit, maxShipSize);
        System.out.println("pre pruning options: " + potentialGuesses.size());
        potentialGuesses = isViableOption(potentialGuesses, maxShipSize, game, pastGuesses);
        // if(potentialGuesses.size() == 0){
        //     System.out.println("[Player1] Protocol: Desperation");
        //     potentialGuesses = populateMatrixAlt(yLimit, xLimit, maxShipSize);
        //     potentialGuesses = isViableOption(potentialGuesses, biggestShip, game, pastGuesses);
        // }
        System.out.println("post pruning options: " + potentialGuesses.size());
        int index = rand.nextInt(potentialGuesses.size());
        guess = potentialGuesses.get(index);
        turnCount++;
        pastGuesses.add(guess);
        return guess;
    }



    @Override
    public void afterGameEnds(final GameView game) {}

}
