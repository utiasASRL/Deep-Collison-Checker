#include "sdfstring.hh"
#include <sstream> 
#include <iostream>

/*
SDFString::SDFString(){
    this->value = 
    "<sdf version ='1.6'>\
    </sdf>";
}
*/

SDFTag::SDFTag(std::string _name){
    
    this->name = _name;
}

std::string SDFTag::WriteTag(int pretabs){
    return "";
}


DataTag::DataTag(std::string _name, std::string _data):SDFTag(_name){
    this->data = _data;
}

std::string DataTag::WriteTag(int pretabs){
    std::stringstream tag;

    for (int i = 0; i<pretabs; i++){
        tag << "\t";
    }

    tag << "<" << this->name << ">" << this->data << "</" << this->name << ">\n";

    return tag.str();
    
}



void HeaderTag::AddAttribute(std::string title, std::string value){
    std::vector<std::string> new_attribute;
    new_attribute.push_back(title);
    new_attribute.push_back(value);

    this->attributes.push_back(new_attribute);
}

void HeaderTag::AddSubtag(std::shared_ptr<SDFTag> _tag){
    this->sub_tags.push_back(_tag);
}

std::string HeaderTag::WriteTag(int pretabs){
    std::stringstream tag;

    for (int i = 0; i<pretabs; i++){
        tag << "\t";
    }

    tag << "<" << this->name;
    

    for (std::vector<std::string> attribute: this->attributes){
        tag << " " << attribute[0] << "=\"" << attribute[1] << "\"";
    }
   
    tag << ">\n";

    for (std::shared_ptr<SDFTag> sub_tag: this->sub_tags){
        tag <<  sub_tag->WriteTag(pretabs+1);
    }
   
    for (int i = 0; i<pretabs; i++){
        tag << "\t";
    }

    tag << "</" << this->name << ">\n";

    return tag.str();
    
}


SDFPlugin::SDFPlugin(std::string _name, std::string _filename): HeaderTag("plugin"){
    this->name = _name;
    this->filename = _filename;
    this->AddAttribute("name", _name);
    this->AddAttribute("filename", _filename);
}

void SDFPlugin::AddSubtag(std::string name ,std::string value){
    std::shared_ptr<DataTag> data = std::make_shared<DataTag>(name, value);
    HeaderTag::AddSubtag(data);
}

SDFAnimation::SDFAnimation(std::string _name ,std::string _filename, bool _interpolate_x): HeaderTag("animation"){
    this->name = _name;
    this->filename = _filename;
    this->interpolate_x = _interpolate_x;
    this->AddAttribute("name", _name);
    std::shared_ptr<DataTag> file = std::make_shared<DataTag>("filename", _filename);
    this->AddSubtag(file);
    if (_interpolate_x){
        std::shared_ptr<DataTag> inter = std::make_shared<DataTag>("interpolate_x", "true");
        this->AddSubtag(inter);
    } else{
        std::shared_ptr<DataTag> inter = std::make_shared<DataTag>("interpolate_x", "false");
        this->AddSubtag(inter);
    }
}
