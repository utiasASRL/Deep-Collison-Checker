#ifndef SDFSTRING_HH
#define SDFSTRING_HH

#include <string>
#include <vector>
#include <gazebo/gazebo.hh>
#include <utility>


class SDFTag{

    public: 

        std::string name;

        SDFTag(std::string _name);

        virtual std::string WriteTag(int pretabs);

};

class DataTag : public SDFTag{

    public: 

        std::string data;

        DataTag(std::string _name, std::string _data);

        std::string WriteTag(int pretabs);

};



class HeaderTag : public SDFTag{

    private:

        std::vector<std::shared_ptr<SDFTag>> sub_tags;

        std::vector<std::vector<std::string>> attributes;

    public:

        using SDFTag::SDFTag;

        std::string WriteTag(int pretabs);

        void AddAttribute(std::string title, std::string value);

        void AddSubtag(std::shared_ptr<SDFTag> _tag);

};

class SDFPlugin : public HeaderTag{

    public:

        std::string name;

        std::string filename;

        SDFPlugin(std::string _name, std::string _filename);

        void AddSubtag(std::string name, std::string value);

};

class SDFAnimation : public HeaderTag{

    public:

        std::string name;
        std::string filename;
        bool interpolate_x;

        SDFAnimation(std::string _name, std::string _filename, bool _interpolate_x);

};


#endif