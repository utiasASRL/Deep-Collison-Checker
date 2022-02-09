#pragma once 

#include <algorithm>
#include <queue>

template <class T, class priority_t>
class PriorityQueue : public std::priority_queue<std::pair<priority_t, T>, std::vector<std::pair<priority_t, T>>, std::greater<std::pair<priority_t, T>>>
{
	public:

		typedef typename 
			std::priority_queue<
				T, 
				std::vector<std::pair<priority_t, T>>,  
				std::greater<std::pair<priority_t, T>>>::container_type::const_iterator const_iterator;

		const_iterator find(const T&val) const{
            auto first = this->c.cbegin();
            auto last = this->c.cend();
            while (first!=last) {
                if ((*first).second==val) return first;
                ++first;
            }
            return last;
        }

        const_iterator last() const{
            return this->c.cend();
        }

		T get(){
			T best = this->top().second;
			this->pop();
			return best;
		}

        void put(T item, priority_t priority){
		    this->emplace(priority, item);
	    }

        void remove(const T&val){
            auto first = this->c.cbegin();
            auto last = this->c.cend();
            while (first!=last) {
                if ((*first).second==val){
                    this->c.erase(first);
                }
                ++first;
            }
            
        }

        int size(){
            return this->c.size();
        }

        bool clear() {
		    this->c.clear();
	    }

};