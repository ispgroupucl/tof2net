/*
    Limited Memory stack
*/

function LimitedStack(size){
    return {
        "maxSize": size,
        "size": 0,
        "data": [],
        "pop": () => {
            if(!this.size)  return undefined;
            else            return this.data[this.size--];
        },
        "push": (elem) => {
            this.data[this.size++] = elem;
            if (this.size==maxSize) this._deleteOldest();
            return this.size;
        },
        "peek": () => {
            if(size) return this.data[this.size-1];
            return undefined;
        },
        "_deleteOldest": () => {
            this.data = this.data.slice(1);
            this.size--;
        },
        "clear": () => {
            this.size = 0;
            this.data = [];
        }
    }
}
function History(size){
    return {
        "past":   LimitedStack(size),
        "future": LimitedStack(size),
        "add": (elem) => {
            this.past.push(elem);
        },
        "back": () => {
            let elem = this.past.pop();
            if( elem == undefined) return undefined;
            this.future.push(elem);
            return this.past.peek();
        },
        "restore": () => {
            let elem = this.future.pop();
            if( elem == undefined) return undefined;
            this.past.push(elem);
            return elem;
        }
    }
}



/*function LimitedStackLL(size){
    return {
        "maxSize": size,
        "size": 0,
        "data": [],
        "push": (elem) => {
            
        },
        "pop": (elem) => {
            this.data
        },
        "_deleteOldest": () => {
            this.data = this.data.slice(1);
        },
   
    }
}
function Node(value, prev, next){
    return {
        "value": value,
        "prev": prev,
        "next": next
    }
}*/