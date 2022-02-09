#include "quadtree.hh"
#include "utilities.hh"



QTData::QTData(ignition::math::Box box, boost::shared_ptr<void> data, types type){
    this->box = box;
    this->data = data;
    this->type = type;
}

QuadTree::QuadTree(ignition::math::Box _boundary){
    this->boundary = _boundary;
}

void QuadTree::Subdivide(){
    auto corners = utilities::get_box_corners(this->boundary);

    ignition::math::Vector3d bot_l = corners[0];
	ignition::math::Vector3d bot_r = corners[1];
	ignition::math::Vector3d top_l = corners[2];
	ignition::math::Vector3d top_r = corners[3];

    this->top_left = std::make_shared<QuadTree>(ignition::math::Box((bot_l + top_l) /2, (top_l+top_r) /2));
    this->top_right = std::make_shared<QuadTree>(ignition::math::Box((bot_l + top_r) /2, top_r));
    this->bot_left = std::make_shared<QuadTree>(ignition::math::Box(bot_l, (bot_l + top_r) /2));
    this->bot_right = std::make_shared<QuadTree>(ignition::math::Box((bot_l + bot_r) /2, (top_r+bot_r) /2));
}


bool QuadTree::Insert(QTData data){

    auto rect = data.box;

    if (!utilities::contains(this->boundary, rect)){
        return false;
    }

    if (this->bot_left == nullptr && this->objects.size() < this->capacity){ // if I dont have children and 
        this->objects.push_back(data);
        return true;
    }

    if (this->bot_left == nullptr){
        this->Subdivide();
    }


    bool stored_in_child = (bot_left->Insert(data)) || (bot_right->Insert(data)) || (top_left->Insert(data)) || (top_right->Insert(data));

    if (!stored_in_child){
        this->objects.push_back(data);
    }
    
    return true;
}

std::vector<QTData> QuadTree::QueryRange(ignition::math::Box range){

    std::vector<QTData> res;

    if (!this->boundary.Intersects(range)){
        return res;
    }

    for (auto object: this->objects){
        if (object.box.Intersects(range)){
            res.push_back(object);
        }
    }

    if (this->top_left == nullptr){
        return res;
    }


    auto tl = this->top_left->QueryRange(range);
    res.insert( res.end(), tl.begin(), tl.end());

    auto tr = this->top_right->QueryRange(range);
    res.insert( res.end(), tr.begin(), tr.end());

    auto bl = this->bot_left->QueryRange(range);
    res.insert( res.end(), bl.begin(), bl.end());

    auto br = this->bot_right->QueryRange(range);
    res.insert( res.end(), br.begin(), br.end());

    return res;

}

void QuadTree::Print(){
    std::printf("boundary box: (%f, %f) -> (%f, %f)\n", this->boundary.Min().X(), this->boundary.Min().Y(), this->boundary.Max().X(), this->boundary.Max().Y());
    for (auto object: this->objects){
        auto box = object.box;
        std::printf("\tobject box: (%f, %f) -> (%f, %f)\n", box.Min().X(), box.Min().Y(), box.Max().X(), box.Max().Y());
    }

    if (this->bot_left == nullptr){
        return;
    }else{
        bot_left->Print();
        bot_right->Print();
        top_left->Print();
        top_right->Print();
    }
}