package src.labs.stealth.agents;

// SYSTEM IMPORTS
import edu.bu.labs.stealth.agents.MazeAgent;
import edu.bu.labs.stealth.graph.Vertex;
import edu.bu.labs.stealth.graph.Path;


import edu.cwru.sepia.environment.model.state.State.StateView;


import java.util.HashSet;       // will need for bfs
import java.util.Queue;         // will need for bfs
import java.util.LinkedList;    // will need for bfs
import java.util.Set;           // will need for bfs
import java.util.HashMap;       // me own treasure
import java.util.ArrayList;     // krust krab
import java.util.Stack;


// JAVA PROJECT IMPORTS


public class BFSMazeAgent
    extends MazeAgent
{

    public BFSMazeAgent(int playerNum)
    {
        super(playerNum);
    }

    //helper method to get neighbors of a node
    public ArrayList<Vertex> getNeighbors(Vertex v, StateView state){
        ArrayList<Vertex> neighbors = new ArrayList<Vertex>();
        int x = v.getXCoordinate();
        int y = v.getYCoordinate();
        if(state.inBounds(x + 1, y) && !state.isResourceAt(x + 1, y)){
            neighbors.add(new Vertex(x + 1, y));
        }
        if(state.inBounds(x - 1, y) && !state.isResourceAt(x - 1, y)){
            neighbors.add(new Vertex(x - 1, y));
        }
        if(state.inBounds(x, y + 1) && !state.isResourceAt(x, y + 1)){
            neighbors.add(new Vertex(x, y + 1));
        }
        if(state.inBounds(x, y - 1) && !state.isResourceAt(x, y - 1)){
            neighbors.add(new Vertex(x, y - 1));
        }
        if(state.inBounds(x + 1, y + 1) && !state.isResourceAt(x + 1, y + 1)){
            neighbors.add(new Vertex(x + 1, y + 1));
        }
        if(state.inBounds(x - 1, y - 1) && !state.isResourceAt(x - 1, y - 1)){
            neighbors.add(new Vertex(x - 1, y - 1));
        }
        if(state.inBounds(x + 1, y - 1) && !state.isResourceAt(x + 1, y - 1)){
            neighbors.add(new Vertex(x + 1, y - 1));
        }
        if(state.inBounds(x - 1, y + 1) && !state.isResourceAt(x - 1, y + 1)){
            neighbors.add(new Vertex(x - 1, y + 1));
        }
        return neighbors;
    }

    @Override
    public Path search(Vertex src,
                       Vertex goal,
                       StateView state)
    {
        Vertex cur = src;
        Queue<Vertex> to_visit = new LinkedList<Vertex>();
        HashSet<Vertex> visited = new HashSet<Vertex>();
        HashMap<Vertex,Vertex> brainmap = new HashMap<Vertex,Vertex>();
        // brainmap.put(src, null);
        while (cur != null){
            visited.add(cur);
            if(to_visit.contains(cur)){
                to_visit.remove(cur);
            }
            for(Vertex v : getNeighbors(cur, state)){
                if(!visited.contains(v) && !to_visit.contains(v)){
                    to_visit.add(v);
                    brainmap.put(v, cur);
                }
            }
            cur = to_visit.poll();
        }
        // System.out.println("visited");
        // System.out.println(visited);
        //time to make the path and use the data retrieved from bfs
        Vertex temp = goal;
        ArrayList<Vertex> toMakePath = new ArrayList<Vertex>();
        Path path = new Path(src);
        // System.out.println(brainmap);
        while(brainmap.get(temp) != src){
            toMakePath.add(brainmap.get(temp));
            temp = brainmap.get(temp);
        }
        // System.out.println("tomakepath\n");
        // System.out.println(toMakePath);
        for(int i = toMakePath.size() - 1; i >= 0; i--){
            if(toMakePath.get(i) == null){
                break;
            }
            path = new Path(toMakePath.get(i), 1f, path);
        }
        // System.out.println(path.toString());
        return path;
    }

    @Override
    public boolean shouldReplacePlan(StateView state)
    {
        Stack<Vertex> plan = BFSMazeAgent.super.getCurrentPlan();
        for(Vertex v : plan){
            if(state.isUnitAt(v.getXCoordinate(), v.getYCoordinate()) && state.isResourceAt(v.getXCoordinate(), v.getYCoordinate())){
                return true;
            }
        }
        return false;
    }

}
