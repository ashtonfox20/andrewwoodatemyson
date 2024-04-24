package src.labs.stealth.agents;

// SYSTEM IMPORTS
import edu.bu.labs.stealth.agents.MazeAgent;
import edu.bu.labs.stealth.graph.Vertex;
import edu.bu.labs.stealth.graph.Path;


import edu.cwru.sepia.environment.model.state.State.StateView;

import java.util.ArrayList;
import java.util.HashSet;   // will need for dfs
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;     // will need for dfs
import java.util.Set;       // will need for dfs
import java.util.HashMap;       // me own treasure
import java.util.ArrayList;     // krust krab
import java.util.Stack;


// JAVA PROJECT IMPORTS


public class DFSMazeAgent
    extends MazeAgent
{

    public DFSMazeAgent(int playerNum)
    {
        super(playerNum);
    }

    public Vertex getNeighbor(Vertex v, StateView state, HashSet<Vertex> visited){
        // ArrayList<Vertex> neighbors = new ArrayList<Vertex>();
        int x = v.getXCoordinate();
        int y = v.getYCoordinate();
        if(state.inBounds(x + 1, y) && !state.isResourceAt(x+1,y) && !visited.contains(new Vertex(x + 1, y))){
            return new Vertex(x + 1, y);
        }
        if(state.inBounds(x - 1, y) && !state.isResourceAt(x-1,y) && !visited.contains(new Vertex(x - 1, y))){
            return new Vertex(x - 1, y);
        }
        if(state.inBounds(x, y + 1) && !state.isResourceAt(x,y+1) && !visited.contains(new Vertex(x, y + 1))){
            return new Vertex(x, y+1);
        }
        if(state.inBounds(x, y - 1) && !state.isResourceAt(x,y-1) && !visited.contains(new Vertex(x, y - 1))){
            return new Vertex(x, y-1);
        }
        if(state.inBounds(x + 1, y + 1) && !state.isResourceAt(x+1,y+1) && !visited.contains(new Vertex(x + 1, y + 1))){
            return new Vertex(x + 1, y+1);
        }
        if(state.inBounds(x - 1, y - 1) && !state.isResourceAt(x-1,y-1) && !visited.contains(new Vertex(x - 1, y - 1))){
            return new Vertex(x - 1, y-1);
        }
        if(state.inBounds(x + 1, y - 1) && !state.isResourceAt(x+1,y-1) && !visited.contains(new Vertex(x + 1, y - 1))){
            return new Vertex(x + 1, y-1);
        }
        if(state.inBounds(x - 1, y + 1) && !state.isResourceAt(x-1,y+1) && !visited.contains(new Vertex(x - 1, y + 1))){
            return new Vertex(x - 1, y+1);
        }
        return null;
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
        while (cur != null){
            visited.add(cur);
            if(to_visit.contains(cur)){
                to_visit.remove(cur);
            }
            Vertex v = getNeighbor(cur, state, visited);
            if(v != null){
                to_visit.add(v);
                brainmap.put(v, cur);
                cur = to_visit.poll();
            }else{
                cur = brainmap.get(cur);
                continue;
            }
            // for(Vertex v : getNeighbor(cur, state, visited)){
            //     if(!visited.contains(v) && !to_visit.contains(v)){
            //         to_visit.add(v);
            //         brainmap.put(v, cur);
            //     }
            // }
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
        Stack<Vertex> plan = DFSMazeAgent.super.getCurrentPlan();
        for(Vertex v : plan){
            if(state.isUnitAt(v.getXCoordinate(), v.getYCoordinate()) && state.isResourceAt(v.getXCoordinate(), v.getYCoordinate())){
                return true;
            }
        }
        return false;
    }

}
