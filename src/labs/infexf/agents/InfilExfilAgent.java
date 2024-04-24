package src.labs.infexf.agents;

import java.util.Set;
import java.util.Stack;
import java.util.Iterator;

// SYSTEM IMPORTS
import edu.bu.labs.infexf.agents.SpecOpsAgent;
import edu.bu.labs.infexf.distance.DistanceMetric;
import edu.bu.labs.infexf.graph.Vertex;
import edu.bu.labs.infexf.graph.Path;


import edu.cwru.sepia.environment.model.state.State.StateView;


// JAVA PROJECT IMPORTS


public class InfilExfilAgent
    extends SpecOpsAgent
{

    public InfilExfilAgent(int playerNum)
    {
        super(playerNum);
    }

    // if you want to get attack-radius of an enemy, you can do so through the enemy unit's UnitView
    // Every unit is constructed from an xml schema for that unit's type.
    // We can lookup the "range" of the unit using the following line of code (assuming we know the id):
    //     int attackRadius = state.getUnit(enemyUnitID).getTemplateView().getRange();
    @Override
    public float getEdgeWeight(Vertex src,
                               Vertex dst,
                               StateView state)
    {
        int dstX = dst.getXCoordinate();
        int dstY = dst.getYCoordinate(); //just some initializations to make the code cleaner
        //For now, I cant think of why we would need to use the src coords, but ill keep them initialized just in case
        double dangerscore = 0;
        Set<Integer> enemies = InfilExfilAgent.super.getOtherEnemyUnitIDs();
        for(Integer enemyId: enemies){
            int enemyX = state.getUnit(enemyId).getXPosition();
            int enemyY = state.getUnit(enemyId).getYPosition(); //a little more initialization for cleanliness            

            //eudclidean distance
            if(dangerscore == 0){ //yeah
                dangerscore += Math.sqrt(Math.pow(enemyX - dstX, 2) + Math.pow(enemyY - dstY, 2));
            }
            if(dangerscore > Math.sqrt(Math.pow(enemyX - dstX, 2) + Math.pow(enemyY - dstY, 2))){
                dangerscore = Math.sqrt(Math.pow(enemyX - dstX, 2) + Math.pow(enemyY - dstY, 2));
            }
        }
        dangerscore = 100000 * Math.pow((1 - 0.7), dangerscore);
        return (float)dangerscore;
    }

    @Override
    public boolean shouldReplacePlan(StateView state)
    {//my methodology for this is going to look at the next three steps in the path, no further, and check for enemy overlap
        Set<Integer> enemies = InfilExfilAgent.super.getOtherEnemyUnitIDs();
        Stack<Vertex> path = InfilExfilAgent.super.getCurrentPlan();

        Iterator<Vertex> iter = path.iterator();
        for(int i = 0; i < 3; i++){
            if(!iter.hasNext()) {
                break;
            }
            Vertex v = iter.next();
            for(Integer enemyId: enemies){
                if (state.getUnit(enemyId) == null){
                    continue;
                }
                int attackRadius = state.getUnit(enemyId).getTemplateView().getRange();
                int enemyX = state.getUnit(enemyId).getXPosition();
                int enemyY = state.getUnit(enemyId).getYPosition();
                if(Math.abs(enemyX - v.getXCoordinate()) <= attackRadius+3 && Math.abs(enemyY - v.getYCoordinate()) <= attackRadius+3){
                    return true;
                }
            }
        }
    return false;
    }

}
