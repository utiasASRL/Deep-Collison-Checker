#include "costmap.hh"

Costmap::Costmap(ignition::math::Box boundary, double resolution){
    this->boundary = boundary;
    this->resolution = resolution;

    this->top_left = ignition::math::Vector3d(boundary.Min().X(), boundary.Max().Y(), 0);
    this->width = boundary.Max().X() - boundary.Min().X();
    this->height = boundary.Max().Y() - boundary.Min().Y();

    this->cols = this->width/this->resolution;
    this->rows = this->height/this->resolution;
    
    for (int r = 0; r <this->rows; ++r){
        std::vector<int> new_row;
        for (int c = 0; c< this->cols; ++c){
            new_row.push_back(1);
        }
        this->costmap.push_back(new_row);
    }

    this->last_path = this->costmap;
    this->obj_count = 0;

}

void Costmap::AddObject(ignition::math::Box object){
    object.Min().Z() = 0;
    object.Max().Z() = 0;

    auto tl = ignition::math::Vector3d(object.Min().X(), object.Max().Y(), 0);
    auto br = ignition::math::Vector3d(object.Max().X(), object.Min().Y(), 0);

    if (tl.X() >= this->boundary.Max().X() || tl.Y() <= this->boundary.Min().Y() || br.X() <= this->boundary.Min().X() || br.Y() >= this->boundary.Max().Y()){
        return;
    }

    int min_r, min_c;
    int max_r, max_c;

    this->PosToIndicies(tl, min_r, min_c);
    
    this->PosToIndicies(br, max_r, max_c);

    //std::printf("tl: (%f, %f), br: (%f, %f), min: (%d, %d), max (%d, %d)\n", tl.X(), tl.Y(), br.X(), br.Y(), min_r, min_c, max_r, max_c);

    for (int r = min_r; r<=max_r; ++r){
        for(int c = min_c; c<=max_c; ++c){
            this->costmap[r][c] = 255;
        }
    }

    this->last_path = this->costmap;
    this->obj_count++;

}

std::string Costmap::ToString(){
    std::stringstream out;

    for (int r = 0; r<this->rows; r++){
        for (int c= 0; c<this->cols; c++){
            if (this->costmap[r][c] == 1){
                out << "*";
            } else{
                out << "▇";
            }
        }
        out << "\n";
    }


    return out.str();
}


std::string Costmap::PathString(std::vector<TrajPoint> path){

    this->last_path = this->costmap;
    std::stringstream out;

    for (auto pt: path){
        int r,c;
        this->PosToIndicies(pt.pose.Pos(), r, c);
        this->last_path[r][c] = -1;
    }

    

    for (int r = 0; r<this->rows; r++){
        for (int c= 0; c<this->cols; c++){
            if (this->last_path[r][c] == 1){
                out << " ";
            } else if (this->last_path[r][c] == -1) {
                out << "X";
            } else {
                out << "▇";
            }
        }
        out << "\n";
    }

    return out.str();
}


bool Costmap::Walkable(ignition::math::Vector3d start, ignition::math::Vector3d end){
    // sample points every 1/5th of resolution along the line and check if it is in an occupied cell.

    auto dir = end-start;
    double length = dir.Length();
    int num = 10;
    int N = (int) length/(this->resolution/num);
    dir = dir.Normalize();
    dir*= this->resolution/num;

    for (int i =1; i <= N; i++){
        auto check_point = dir*i + start;
        int r,c;
        this->PosToIndicies(check_point, r, c);

        if (this->costmap[r][c] != 1){
            return false;
        }
    }

    return true;
}


std::vector<std::vector<int>> Costmap::GetNeighbours(std::vector<int> curr_ind, bool diag){
    std::vector<std::vector<int>> res;

    if (curr_ind[0] > 0){ // we can return top
        res.push_back({curr_ind[0]-1, curr_ind[1]});
    }

    if (curr_ind[1] > 0){ // we can return left
        res.push_back({curr_ind[0], curr_ind[1]-1});
    }

    if (curr_ind[0] < this->rows-1){ // we can return bot
        res.push_back({curr_ind[0]+1, curr_ind[1]});
    }

    if (curr_ind[1] < this->cols-1){ // we can return right
        res.push_back({curr_ind[0], curr_ind[1]+1});
    }

    if (diag){
        if (curr_ind[0] > 0 && curr_ind[1] > 0){ // we can return top left
            res.push_back({curr_ind[0]-1,curr_ind[1]-1});
        }

        if (curr_ind[0] > 0 && curr_ind[1] <this->cols-1){ // we can return bottom left
            res.push_back({curr_ind[0]-1,curr_ind[1]+1});
        }

        if (curr_ind[0] < this->rows-1 && curr_ind[1] > 0){ // we can return top right
            res.push_back({curr_ind[0]+1,curr_ind[1]-1});
        }

        if (curr_ind[0] < this->rows-1 && curr_ind[1] < this->cols-1){ // we can return bottom right
            res.push_back({curr_ind[0]+1,curr_ind[1]+1});
        }
    }

    return res;
}

/*
* GetNeighbourAngle returns the angle in degres between the current grid position and a given neighbouring positions. 
* Works on the [x_min-1, x_max-1][y_min-1, y_max-1] range of the grid. Intended to use for Flow Field generation.
*
* @param curr_ind current grid indices
* @param neighbour_ind neighbour grid indices
*/
double Costmap::GetNeighbourAngle(std::vector<int> curr_ind, std::vector<int> neighbour_ind){
    static const double TWOPI = 6.2831853071795865;
    static const double RAD2DEG = 57.2957795130823209;

    if (curr_ind == neighbour_ind){
        return 0.0;
    }

    double theta = std::atan2(neighbour_ind[0] - curr_ind[0], curr_ind[1] - neighbour_ind[1]);
    if(theta<0.0){
        theta += TWOPI;
    }
    // std::cout << "Angle in deg: " << RAD2DEG * theta << " between positions: " << curr_ind[0] << " " << curr_ind[1] << " and " << neighbour_ind[0] << " " << neighbour_ind[1] << std::endl;
    return RAD2DEG * theta;
}

bool Costmap::PosToIndicies(ignition::math::Vector3d pos, int &r, int &c){
    
    r = (int)floor((top_left.Y() - pos.Y()) / resolution);
    c = (int)floor((pos.X() - top_left.X()) / resolution);

    /*
    int r0 = 0;
    int c0 = 0;
    while ((this->top_left.Y() - r*this->resolution - this->resolution) > pos.Y()){
        r++;
    }

    while ((this->top_left.X() + c*this->resolution + this->resolution) < pos.X()){
        c++;
    }
    if (r != r0 || c != c0)
    {
        std::cout << "----------------------------------- " << std::endl;
        std::cout << "---------------------> r0 = " << r0 << " but r = " << r << "  / max = " << rows << std::endl;
        std::cout << "---------------------> c0 = " << c0 << " but c = " << c << "  / max = " << cols << std::endl;
        std::cout << "----------------------------------- " << std::endl;
    }
    */

    return utilities::inside_box(this->boundary, pos, true);
}

bool Costmap::IndiciesToPos(ignition::math::Vector3d &pos, int r, int c){

    pos = ignition::math::Vector3d(this->boundary.Min().X() + c*this->resolution, this->boundary.Max().Y() - r*this->resolution, 0);
    return ((r>=0 && r < this->rows) && (c>=0 && c < this->cols));
}

double Costmap::Heuristic(std::vector<int> loc1, std::vector<int> loc2){
    ignition::math::Vector3d pos1, pos2;
    this->IndiciesToPos(pos1, loc1[0],loc1[1]);
    this->IndiciesToPos(pos2, loc2[0],loc2[1]);

    return (pos1-pos2).Length();
}

bool Costmap::Occupied(ignition::math::Vector3d pos){
    int r,c;
    this->PosToIndicies(pos, r, c);
    
    return (this->costmap[r][c] > 1);
}

ignition::math::Vector3d Costmap::RandPos(){
    // select random height
    int rand_row = ignition::math::Rand::IntUniform(0, this->rows-1);

    // work across and save safe spaces (those that are after only one collision)

    
    bool found = false;
    int count = 0;
    
    while (!found && count < 1000){
        
        std::vector<int> collisions;

        for (int c = 0; c < this->cols; c++){

            bool occupied = (this->costmap[rand_row][c] >1);

            if (occupied){

                if (c == 0){
                    if (this->costmap[rand_row][1] == 1){
                        collisions.push_back(c);
                        collisions.push_back(c);
                    }else{
                        collisions.push_back(c);
                    }
                } else if (c == this->cols-1){
                    if (this->costmap[rand_row][c-1] == 1){
                        collisions.push_back(c);
                        collisions.push_back(c);
                    }else{
                        collisions.push_back(c);
                    }
                } else{
                    if (this->costmap[rand_row][c-1] == 1 && this->costmap[rand_row][c+1] == 1){
                        collisions.push_back(c);
                        collisions.push_back(c);
                    } else if (this->costmap[rand_row][c-1] == 1 || this->costmap[rand_row][c+1] == 1){
                        collisions.push_back(c);
                    }
                }
            }


        }

        std::vector<int> safe_cols;

        for (int ind = 1; ind < collisions.size(); ind+=2){
            int start_c = collisions[ind];
            int end_c = this->cols;
            if (ind+1 < collisions.size()){
                end_c = collisions[ind+1];
            }

            for (int i = start_c+1; i < end_c; i++){
                safe_cols.push_back(i);
            }
        
        }

        if (safe_cols.size() > 0){
            found = true;
        } else {
            count ++;
            continue;
        }

        int rand_ind = ignition::math::Rand::IntUniform(0,safe_cols.size()-1);

        ignition::math::Vector3d res;
        this->IndiciesToPos(res, rand_row, safe_cols[rand_ind]);

        return res;
    }

    std::cout << "Failed to find random target for actor\n" << std::endl;
    return ignition::math::Vector3d(0,0,0);
}

bool Costmap::AStar(ignition::math::Vector3d start, ignition::math::Vector3d end, std::vector<ignition::math::Vector3d> &path, bool straighten){
    auto t1 = std::chrono::high_resolution_clock::now();

    this->parent.clear();
    this->g_cost.clear();
    this->open.clear();
    
    int start_r, start_c, end_r, end_c;
    this->PosToIndicies(start, start_r, start_c);
    this->PosToIndicies(end, end_r, end_c);


    std::vector<int> start_coords = {start_r, start_c};
    std::vector<int> end_coords = {end_r, end_c};
    //std::cout << "end: ";
    
    this->target = end_coords;

    this->g_cost[start_coords] = 0;
    this->parent[start_coords] = start_coords;
    
    this->open.put(start_coords, this->Heuristic(start_coords, end_coords));
    std::set<std::vector<int>> closed;

    bool found = false;

    while (!this->open.empty()){
        //std::cout << open.size() << std::endl;
        
        auto s = this->open.get();
        //print_coords(s);
        if (s[0] == end_coords[0] && s[1] == end_coords[1]){
            found = true;
            break; // path found
        }

        closed.insert(s);

        for (auto n: this->GetNeighbours(s, true)){
            double n_cost = this->costmap[n[0]][n[1]];
            if (n_cost > 1){ // if we encounter a wall, skip 
                continue;
            }
            if (closed.find(n) == closed.end()){
                if (this->open.find(n) == open.last()){
                    this->g_cost[n] = std::numeric_limits<double>::infinity();
                }
                this->UpdateVertexA(s, n);
            }
        }
    }


    if (!found){
        std::cout << "No Path Found\n";
        return false;
    }


    auto curr_coords = end_coords;
    ignition::math::Vector3d actual_pos = end;
    ignition::math::Vector3d last_pos = end;

    int count = 0;

    while (curr_coords[0] != start_coords[0] || curr_coords[1] != start_coords[1]){

        if (count != 0){
            ignition::math::Vector3d curr_pos;
            this->IndiciesToPos(curr_pos, curr_coords[0], curr_coords[1]);
            auto offset = curr_pos - last_pos;
            actual_pos = actual_pos+offset;
            path.push_back(actual_pos);
        } else{
            path.push_back(end);
        }

        this->IndiciesToPos(last_pos, curr_coords[0], curr_coords[1]);
        curr_coords = this->parent[curr_coords];
        

        count ++;
    }

    path.push_back(start);
    std::reverse(path.begin(), path.end());

    if (straighten){
        int check_ind = 0;
        int next_ind =1;
        while (next_ind < path.size()-1){
            if (this->Walkable(path[check_ind],path[next_ind+1])){
                path.erase(path.begin()+next_ind);
            } else{
                check_ind = next_ind;
                next_ind = check_ind +1;
            }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout <<  "Path Found (A*). Duration: " << ((double)duration)/10e6 << " s"<< std::endl;
    return true;
}

void Costmap::UpdateVertexA(std::vector<int> s, std::vector<int> n){
    auto c = this->DistCost(s,n);
    if (this->g_cost[s] + c < this->g_cost[n]){
        this->g_cost[n] = this->g_cost[s] + c;
        this->parent[n] = s;
        if (this->open.find(n) != this->open.last()){
            this->open.remove(n);
        }
        this->open.put(n, this->g_cost[n] + this->Heuristic(n, this->target));
    }
}

bool Costmap::ThetaStar(ignition::math::Vector3d start, ignition::math::Vector3d end, std::vector<ignition::math::Vector3d> &path){
    auto t1 = std::chrono::high_resolution_clock::now();

    this->parent.clear();
    this->g_cost.clear();
    this->open.clear();
    
    int start_r, start_c, end_r, end_c;
    this->PosToIndicies(start, start_r, start_c);
    this->PosToIndicies(end, end_r, end_c);


    std::vector<int> start_coords = {start_r, start_c};
    std::vector<int> end_coords = {end_r, end_c};
    //std::cout << "end: ";
    
    this->target = end_coords;

    this->g_cost[start_coords] = 0;
    this->parent[start_coords] = start_coords;
    
    this->open.put(start_coords, this->Heuristic(start_coords, end_coords));
    std::set<std::vector<int>> closed;

    bool found = false;

    while (!this->open.empty()){
        //std::cout << open.size() << std::endl;
        
        auto s = this->open.get();
        //print_coords(s);
        if (s[0] == end_coords[0] && s[1] == end_coords[1]){
            found = true;
            break; // path found
        }

        closed.insert(s);

        for (auto n: this->GetNeighbours(s, true)){
            double n_cost = this->costmap[n[0]][n[1]];
            if (n_cost > 1){ // if we encounter a wall, skip 
                continue;
            }
            if (closed.find(n) == closed.end()){
                if (this->open.find(n) == open.last()){
                    this->g_cost[n] = std::numeric_limits<double>::infinity();
                }
                this->UpdateVertexB(s, n);
            }
        }
    }


    if (!found){
        return false;
    }


    auto curr_coords = end_coords;
    ignition::math::Vector3d actual_pos = end;
    ignition::math::Vector3d last_pos = end;

    int count = 0;

    while (curr_coords[0] != start_coords[0] || curr_coords[1] != start_coords[1]){

        if (count != 0){
            ignition::math::Vector3d curr_pos;
            this->IndiciesToPos(curr_pos, curr_coords[0], curr_coords[1]);
            auto offset = curr_pos - last_pos;
            actual_pos = actual_pos+offset;
            path.push_back(actual_pos);
        } else{
            path.push_back(end);
        }

        this->IndiciesToPos(last_pos, curr_coords[0], curr_coords[1]);
        curr_coords = this->parent[curr_coords];
        

        count ++;
    }

    path.push_back(start);
    std::reverse(path.begin(), path.end());

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout <<  "Path Found (Theta*). Duration: " << ((double)duration)/10e6 << " s"<< std::endl;
    return true;
}

void Costmap::UpdateVertexB(std::vector<int> s, std::vector<int> n){

    ignition::math::Vector3d p1, p2;

    auto par = this->parent[s];

    this->IndiciesToPos(p1, par[0], par[1]);
    this->IndiciesToPos(p2, n[0], n[1]);

    if (this->Walkable(p1, p2)){


        auto c = this->DistCost(par, n);
        if (this->g_cost[par] + c < this->g_cost[n]){
            this->g_cost[n] = this->g_cost[par]+c;
            this->parent[n] = par;
            if (this->open.find(n) != this->open.last()){
                this->open.remove(n);
            }
            this->open.put(n, this->g_cost[n]+this->Heuristic(n, this->target));
        }
    } else{
        auto c = this->DistCost(s, n);
        if (this->g_cost[s] + c < this->g_cost[n]){
            this->g_cost[n] = this->g_cost[s]+c;
            this->parent[n] = s;
            if (this->open.find(n) != this->open.last()){
                this->open.remove(n);
            }
            this->open.put(n, this->g_cost[n] + this->Heuristic(n, this->target));
        }
    }
}

double Costmap::DistCost(std::vector<int> s, std::vector<int> n){
    ignition::math::Vector3d s_pos;
    ignition::math::Vector3d n_pos;
    this->IndiciesToPos(s_pos, s[0], s[1]);
    this->IndiciesToPos(n_pos, n[0], n[1]);

    return (s_pos-n_pos).Length();
}