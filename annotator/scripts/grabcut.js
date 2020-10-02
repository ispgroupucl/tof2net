/*
  -- HELPER FX --
*/
function uint82Hex(x){
  let hex = x.toString(16);
  return hex.length==1? '0'+hex: hex;
}
function rgb2Hex(rgb){
  return '#' + uint82Hex(rgb[0]) + uint82Hex(rgb[1]) + uint82Hex(rgb[2])
}
function list2Rgb(li){
  return new cv.Scalar(li[0], li[1], li[2]);
}

/*
  -- GLOBAL VARS --
*/

// Global State
var scale = 1.0;
let allFiles = new Set();
let allFilesOrdered = [];
// Utils
let print = console.log;
const DEBUG = true;
let print_info = function(str){ // TODO: change this to a string in some div
  if (DEBUG) {
    print(str);
  }
}
var drawingConstants = undefined;
var drawingTransforms = undefined;
var drawings = undefined; // Contains the currently loaded drawing
var drawingsByClass = undefined;
var colors = undefined;

// Waits for the OpenCV library to be loaded
function onOpenCvReady() {
  document.getElementById('status').innerHTML = 'OpenCV.js is ready.';
  document.getElementById('cv_body').hidden = false;
  // Drawings
  colors = {
    'blue': new cv.Scalar(255,1,1),
    'red' : new cv.Scalar(1,1,255),
    'green':new cv.Scalar(1,255,1),
    'black':new cv.Scalar(1,1,1),
    'white':new cv.Scalar(255,255,255),
    'erase':new cv.Scalar(0,0,0) 
  };
  drawingConstants = {
    'bG': {
      'index': 0,
      'color': colors.black,
      'val':   new cv.Scalar(0)
    },
    'fG': {
      'index': 1,
      'color': colors.white,
      'val':   new cv.Scalar(1)
    },
    'bGProb': {
      'index': 2,
      'color': colors.red,
      'val':   new cv.Scalar(2)
    },
    'fGProb': {
      'index': 3,
      'color': colors.green,
      'val':   new cv.Scalar(3)
    },
    'eraser': {
      'index': 4,
      'color': colors.erase,
      'val': new cv.Scalar(4)
    }
  };
  drawingTransforms = {};
  for (let key in drawingConstants){
    drawingTransforms[drawingConstants[key].val[0]] = drawingTransforms[drawingConstants[key].color]
  }
  let imgCanvas = document.getElementById('canvas_img');
  let outCanvas = document.getElementById('canvas_out');
  drawings = Drawings(imgCanvas, outCanvas);
  imgCanvas.oncontextmenu = function (e) {
      e.preventDefault();
  };
}


/**
  -- Image Show functions
 */

 function showImage(canvas, image) {
  // let scale = 3.0;
  let shape = image.size()
  let image_show = new cv.Mat.zeros(shape.height*scale, shape.width*scale, cv.CV_8UC3);
  print(image_show.size())
  let no_size = {'height':0, 'width':0};
  cv.resize(image, image_show, no_size, scale, scale, cv.INTER_NEAREST);
  cv.imshow(canvas, image_show);
  image_show.delete();
 }

 function updateScale(newScale) {
   print(newScale.valueAsNumber);
   scale = newScale.valueAsNumber;
   drawings.updateAll(); drawingsByClass.updateFused();
 }



/**
  -- Change Thickness --
*/

function updateThickness(newThickness) {
  thickness = newThickness.valueAsNumber;
  drawingsByClass.setThickness(thickness);
}

/*
  -- Img & Classes Loading functions  --
*/
// Loads the selected images into the file-list
async function addImages(){
  print(document.getElementById('file_select'))
  let fileBrowser = document.getElementById('file_select').files;
  let fileList = document.getElementById('file_list').options;
  // TODO add loading animation + async + dynamic resizing of fileList
  for(let i=0; i<fileBrowser.length; ++i){
    let ff = fileBrowser[i];
    if (!allFiles.has(ff)){ // TODO: the selection via set does not work yet
      fileList[fileList.length] = new Option(ff.name, allFiles.size);
      allFiles.add(ff);
      allFilesOrdered = allFilesOrdered.concat(ff);
    }
  }
}

// Reset every selected image + clean canvas
function resetSelection(){
  allFiles.clear();
  allFilesOrdered = [];
  let fileList = document.getElementById('file_list').options;
  fileList.length = 0;
  document.getElementById('invisible_img').src = null;
  const canvas = document.getElementById('canvas_img');
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  drawings.clearMemory();
  console.log("Emptied all files from queue\n");
}

// Load all the classes
function loadAllClasses(){
  let classFiles = document.getElementById("class_file").files;
  if (classFiles.length < 1) return;

  // Remove all old buttons
  let cFile = classFiles[0];
  let classButtons = document.getElementById("class_buttons");
  for(let i=classButtons.childNodes.length-1; i>=0; i--){
    let btn = classButtons.childNodes[i];
    classButtons.removeChild(btn);
  }

  let fr = new FileReader();
  fr.onload = (e) => {
    // Parse file
    let lines = e.target.result;
    let parsedFile = JSON.parse(lines);
    let newButtons = parsedFile.classes;
    let buttonColors = parsedFile.colors;

    // Set Global state
    setNewDrawingsByClass(parsedFile);

    // Add buttons to select class
    let selected = false;
    for (let i in newButtons){
      let lbl = document.createElement("label");
      let btn = document.createElement("input");
      const cl = newButtons[i];
      btn.type = "radio";
      lbl.innerHTML = cl;
      lbl.style.backgroundColor = rgb2Hex(buttonColors[i]);
      btn.name = "classNames";
      btn.id = cl;

      btn.addEventListener("click", () => {
        drawings = drawingsByClass.data[cl];
        drawings.reload();
        console.log("Selected class", cl);
      });
      classButtons.appendChild(lbl);
      classButtons.appendChild(btn);
      if(!selected){
        btn.click();
        selected=true;
      }
    }
    document.getElementById("all_canvasses").hidden = false;
    document.getElementById("all_files").hidden     = false;
  };
  fr.readAsText(cFile);
}


/*
  -- Global State Switcher --
*/
function setNewDrawingsByClass(allClassesAndColors){
  let allClasses = allClassesAndColors.classes;
  let allColors = allClassesAndColors.colors;
  for(let i in allColors) allColors[i] = list2Rgb(allColors[i]);
  
  if (drawingsByClass == undefined){
    drawingsByClass = DrawingsByClass(allClasses, document.getElementById('canvas_img'),
                                        document.getElementById('canvas_out'),
                                        document.getElementById('canvas_fused'));
  } else {
    let newDrawingsByClass = DrawingsByClass([], document.getElementById('canvas_img'),
                                              document.getElementById('canvas_out'),
                                              document.getElementById('canvas_fused'));
    for (let i in allClasses){
      let cl = allClasses[i];
      if (cl in drawingsByClass.data){
        let copiedState = drawingsByClass.removeWithoutDeleting(cl);
        newDrawingsByClass.appendExisting(cl, copiedState);
      } else {
        newDrawingsByClass.appendNew(cl);
      }
    }
    drawingsByClass.delete();
    drawingsByClass = newDrawingsByClass;
  }
  drawingsByClass.colors = allColors;
}

function DrawingsByClass(allClasses, imgCanvas, outCanvas, fusedCanvas){
  let newDrawingsByClass = {
    'data': {},
    'orig': null,
    'imgCanvas': imgCanvas,
    'outCanvas': outCanvas,
    'fusedCanvas': fusedCanvas,
    'name': null, // contains the name of the file actually in memory
    'appendExisting': function(cl, state){
      this.data[cl] = state;
    },
    'appendNew': function(cl){
      this.data[cl] = Drawings(this.imgCanvas, this.outCanvas, this.fusedCanvas);
    },
    'removeWithoutDeleting': function(cl){
      let ret = this.data[cl];
      delete this.data[cl];
      return ret;
    },
    'clone': function(){
      let newDBC = DrawingsByClass([], this.imgCanvas, this.outCanvas, this.fusedCanvas);
      for (let cl in this.data){
        newDBC.data[cl] = this.data[cl].clone()
      }
      newDBC.setOrig(this.orig.clone());
      newDBC.name = this.name;
      return newDBC
    },
    'copyMask': function(){
      newMasks = {};
      for (let cl in this.data){
        newMasks[cl] = this.data[cl].copyMask();
      }
      return newMasks;
    },
    'pasteMask': function(newMasks){
      for (let cl in newMasks){
        this.data[cl].pasteMask(newMasks[cl]);
      }
    },
    'delete': function(){
      for (cl in this.data){
        this.data[cl].delete();
      }
      if (this.orig!==null) this.orig.delete();
    },
    'getFusedResult': function(){
      let ret = new cv.Mat.zeros(drawings.orig.rows, drawings.orig.cols, cv.CV_8UC1);
      let rData = ret.data;
      let clIndex = 1;
      for (let cl in this.data){
        let mat = this.data[cl].mask;
        let mData = mat.data;
        const nRows = mat.rows;
        const nCols = mat.cols;
        for (let i=0; i<nRows; i++){
          for (let j=0; j<nCols; j++){
            const ind = i*nCols + j;
            const mval = mData[ind];
            if (mval==1 || mval==3){
              rData[ind] = clIndex;
            }
          }
        }
        clIndex++;
      }
      return ret;
    },
    'updateFused': function(){
      console.time("fused");
      let fused = this.getFusedResult();
      // TODO: there is a lot of code duplication in this fx sadly
      //        either move to a functional interface for edge computing
      //        & orig img showing
      if(imgShowStatus.annot){
        const nRows = fused.rows;
        const nCols = fused.cols;
        let fused3C = new cv.Mat.zeros(nRows, nCols, cv.CV_8UC3);
        const fData = fused.data;
        const f3CData = fused3C.data;
        const nElems = nRows*nCols;
        let ind3C = 0;
        for (let ind=0; ind<nElems; ++ind){
          const mval = fData[ind];
          if (mval>0){
            const color = this.colors[mval-1];
            f3CData[ind3C++] = color[0];
            f3CData[ind3C++] = color[1];
            f3CData[ind3C++] = color[2];
          } else {
            ind3C = ind3C + 3;
          }
        }
        showImage(this.fusedCanvas, fused3C);
        // cv.imshow(this.fusedCanvas, fused3C);
        fused.delete(); fused3C.delete();
        console.timeEnd("fused");
        return;
        // If you want the masks what's underneath won't matter
      }
      let img = this.orig.clone();
      if (!imgShowStatus.intensity ^ !imgShowStatus.depth){
        const nRows = img.rows;
        const nCols = img.cols;
        const iData = img.data;
        const bad = (!imgShowStatus.depth)? 1: 2;
        const okk = (!imgShowStatus.depth)? 2: 1;
        for (let i=0; i<nRows; ++i){
          for (let j=0; j<nCols; ++j){
            const ind = i*nCols*3 + j*3;
            const value = iData[ind+okk];
            iData[ind] = value;
            iData[ind+bad] = value;
          }
        }
      } else if (!imgShowStatus.intensity && !imgShowStatus.depth){
        img.setTo(new cv.Scalar(128,128,128));
      }

      if(imgShowStatus.annotEdges){
        print("show EDGES");
        // Init object
        const nRows = fused.rows;
        const nCols = fused.cols;
        const nElems = nRows*nCols;
        let annotEdges = new cv.Mat(nRows, nCols, cv.CV_8UC1);
        let annotFilled = new cv.Mat(nRows, nCols, cv.CV_8UC1);
        let colorEdges = new cv.Mat(nRows, nCols, cv.CV_8UC3);
        let annotEdges3C = new cv.Mat.zeros(nRows, nCols, cv.CV_8UC3);
        let kernel = new cv.Mat(3,3,cv.CV_8UC1, new cv.Scalar(1));
        let kData = kernel.data;
        const arr = [0,2];
        for(let i in arr){
          for(let j in arr){
            kData[arr[i]*3+arr[j]] = new cv.Scalar(0);
          }
        }
        for (let cl=1; cl<=this.colors.length; ++cl){
          // Make binary mask          
          annotEdges.setTo(new cv.Scalar(0));
          const fData = fused.data;
          const aData = annotEdges.data;
          for (let ind=0; ind<nElems; ++ind){
            if(fData[ind]==cl) aData[ind] = 1;
          }
          // Find edges
          annotEdges.copyTo(annotFilled);
          cv.erode(annotFilled, annotFilled, kernel);
          cv.subtract(annotEdges, annotFilled, annotEdges);
          // Copy edges over
          colorEdges.setTo(this.colors[cl-1]);
          // annotEdges3C.setTo(cv.Scalar(0,0,0));
          cv.cvtColor(annotEdges, annotEdges3C, cv.COLOR_GRAY2RGB);
          colorEdges.copyTo(img, annotEdges3C);
        }

        annotEdges3C.delete();
        colorEdges.delete();
        annotEdges.delete();
        annotFilled.delete();
        kernel.delete();
      }
      showImage(this.fusedCanvas, img);
      // cv.imshow(this.fusedCanvas, img);
      img.delete(); fused.delete();
      console.timeEnd("fused");
    },
    'setOrig': function(newOrig){
      if (this.orig !== null) this.orig.delete();
      this.orig = newOrig;
      for (let cl in this.data) {
        let draw = this.data[cl];
        draw.orig = newOrig;
        resetDrawings(draw);
      }
    },
    'setThickness': function(newThickness) {
      this.thickness = newThickness;
      for (let cl in this.data) {
        let draw = this.data[cl];
        draw.thickness = newThickness;
      }
    }
  };
  print(newDrawingsByClass)
  for (let i in allClasses){
    let cl = allClasses[i];
    newDrawingsByClass.appendNew(cl);
  }
  return newDrawingsByClass;
}

/*
  -- GrabCut Loop --
*/
function Drawings(imgCanvas, outCanvas){
  return resetDrawings({
    'img': null,
    'annot': null,
    'orig': null,
    'origMask': null,
    'mask': null,
    'output': null, // only used to restore on class change
    'rect': (0,0,1,1),
    'drawing': false,
    'rectangle': false,
    'rectDrawn': false,
    'rectOrMask': 100,
    'drawColors': drawingConstants.fG,
    'thickness': 3,
    'imgCanvas': imgCanvas,
    'outCanvas': outCanvas,
    'clearMemory': function(){
      if (this.img!==null)  this.img.delete();
      if (this.annot!==null) this.annot.delete();
      if (this.mask!==null) this.mask.delete();
      if (this.origMask!==null) this.origMask.delete();
      if (this.output!==null) this.output.delete();
      this.img = this.mask = this.output = this.annot = this.origMask = null;
    },
    'delete': function(){
      this.clearMemory();
    },
    'clone': function(){
      let newD = Drawings(this.imgCanvas, this.outCanvas);
      if (this.img !== null) newD.img = this.img.clone();
      if (this.annot !== null) newD.annot = this.annot.clone();
      if (this.mask !== null) newD.mask = this.mask.clone();
      if (this.origMask !== null) newD.origMask = this.origMask.clone();
      if (this.output !== null) newD.mask = this.output.clone();
      let toCopy = {orig:true, rect:true, drawing:true, rectangle:true, rectDrawn:true,
                    rectOrMask:true, drawColors:true, thickness:true}
      for (let key in this){
        if (key in toCopy){
          newD[key] = this[key];
        }
      }
      return newD;
    },
    'copyMask': function(){
      maskParams = {};
      if (this.origMask !== null) maskParams.origMask = this.origMask.clone();
      if (this.mask !== null) maskParams.mask = this.mask.clone();
      if (this.annot !== null) maskParams.annot = this.annot.clone();
      let toCopy = {rect:true, drawing:true, rectangle:true, rectDrawn:true,
                    rectOrMask:true, drawColors:true}
      for (let key in this){
        if (key in toCopy){
          maskParams[key] = this[key];
        }
      }
      return maskParams;
    },
    'pasteMask': function(maskParams){
      for (let key in maskParams){
        if (key=="origMask" || key=="annot" || key=="mask"){
          if (this[key] !== null) maskParams[key].copyTo(this[key]);
          else                    this[key] = maskParams[key].clone();
        } else {
          this[key] = maskParams[key];
        }
      }
    },
    'updateOutput': function(){
      // console.time("output");
      // Apply mask to orig img to select only segmented part
      let output = this.orig.clone();
      const nRows = output.rows;
      const nCols = output.cols;
      let mData = this.mask.data;
      let oData = output.data;
      for (let i=0; i<nRows; i++){
        for (let j=0; j<nCols; j++){
          const ind = i*nCols + j;
          const mval = mData[ind];
          if (mval==0 || mval==2 || mval==4){
            let ind2 = i*nCols*3 + j*3;
            oData[ind2++] = 255;
            oData[ind2++] = 255;
            oData[ind2++] = 255;
          }
        }
      }
      // Show updated version
      showImage(this.outCanvas, output);
      // cv.imshow(this.outCanvas, output);
      output.delete();
      // console.timeEnd("output");
    },
    'updateImg': function(){
      // console.time("input");
      if (this.img == null) this.img = this.orig.clone();
      else                  this.orig.copyTo(this.img);
      if (!imgShowStatus.intensity ^ !imgShowStatus.depth){
        const nRows = this.img.rows;
        const nCols = this.img.cols;
        const iData = this.img.data;
        const bad = (!imgShowStatus.depth)? 1: 2;
        const okk = (!imgShowStatus.depth)? 2: 1;
        for (let i=0; i<nRows; ++i){
          for (let j=0; j<nCols; ++j){
            const ind = i*nCols*3 + j*3;
            const value = iData[ind+okk];
            iData[ind] = value;
            iData[ind+bad] = value;
          }
        }
      } else if (!imgShowStatus.intensity && !imgShowStatus.depth){
        this.img.setTo(new cv.Scalar(128,128,128));
      }

      if(imgShowStatus.annot) this.annot.copyTo(this.img, this.annot);
      if(imgShowStatus.annotEdges){
        // Make binary mask
        let annotEdges = this.mask.clone();
        const nRows = annotEdges.rows;
        const nCols = annotEdges.cols;
        const aData = annotEdges.data;
        const allVals = [false,true,false,true,false];
        for (let ind=0; ind<nRows*nCols; ++ind){
          const mval = aData[ind];
          if(allVals[mval]) aData[ind] = 1;
          else              aData[ind] = 0;
        }
        // Find edges
        let annotFilled = annotEdges.clone();
        let kernel = new cv.Mat(3,3,cv.CV_8UC1, new cv.Scalar(1));
        let kData = kernel.data;
        const arr = [0,2];
        for(let i in arr){
          for(let j in arr){
            kData[arr[i]*3+arr[j]] = new cv.Scalar(0);
          }
        }
        cv.erode(annotFilled, annotFilled, kernel);
        cv.subtract(annotEdges, annotFilled, annotEdges);
        // Copy edges over (always in red on final img)
        let colorEdges = new cv.Mat(nRows, nCols, cv.CV_8UC3, new cv.Scalar(255,0,0));
        let annotEdges3C = new cv.Mat.zeros(nRows, nCols, cv.CV_8UC3);
        cv.cvtColor(annotEdges, annotEdges3C, cv.COLOR_GRAY2RGB);
        colorEdges.copyTo(this.img, annotEdges3C);

        annotFilled.delete();
        annotEdges.delete();
        annotEdges3C.delete();
        colorEdges.delete();
        kernel.delete();
      }
      showImage(this.imgCanvas, this.img);
      // cv.imshow(this.imgCanvas, this.img);
      // console.timeEnd("input");
    },
    'updateAll': function(){
      this.updateOutput();
      this.updateImg();
    },
    'reload': function(){
      if (this.img !== null){
        this.updateAll();
        updateColorDot();
      }
    },
    'drawRectangle': function(p){
      this.rect = new cv.Rect(Math.min(this.rectStart.x, p.x), Math.min(this.rectStart.y, p.y),
                              Math.abs(this.rectStart.x - p.x), Math.abs(this.rectStart.y - p.y));
      this.annot.delete();
      this.annot = new cv.Mat.zeros(this.img.rows, this.img.cols, cv.CV_8UC3);
      cv.rectangle(this.annot, this.rectStart, p, colors.blue, 2);
      this.updateImg();
    },
    'drawLine': function(p){
      cv.line(this.annot, this.prevLinePoint, p, this.drawColors.color, this.thickness);
      this.updateImg();
      this.updateOutput();
      cv.line(this.mask, this.prevLinePoint, p, this.drawColors.val, this.thickness);
      this.prevLinePoint = p;
    }
  });
}

function resetDrawings(drawings){
  drawings.rect = new cv.Rect(0,0,1,1);
  drawings.drawing = false;
  drawings.rectangle = false;
  drawings.rectOrMask = 100;
  drawings.rectDrawn = false;
  drawings.drawColors = drawingConstants.fG;
  drawings.clearMemory();
  drawings.img = (drawings.orig !== null)? drawings.orig.clone(): null;
  drawings.annot = (drawings.img !== null)? new cv.Mat.zeros(drawings.img.rows, drawings.img.cols, cv.CV_8UC3): null;
  drawings.mask = (drawings.img !== null)? new cv.Mat.zeros(drawings.img.rows, drawings.img.cols, cv.CV_8UC1): null;    // mask initialized to PR_BG
  updateColorDot();
  return drawings;
}


function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
function clearMemory(){
  if (shownImage !== null) shownImage.delete(); 
  if (shownMask !== null) shownMask.delete();
}
function loadIntoCanvas(selectIndex){
  if(selectIndex<0) return;
  // Find img and load it into hidden 
  let fileList = document.getElementById('file_list').options;
  let selectedFile = allFilesOrdered[fileList[selectIndex].value];
  drawingsByClass.name = selectedFile.name;
  let ff = URL.createObjectURL(selectedFile);
  let placeholder = document.getElementById('invisible_img');
  placeholder.src = ff;
  // Wait for img to be loaded in @placeholder before moving it to canvas
  placeholder.addEventListener("load",
    () => {
      // TODO: add loading animation
      let img = cv.imread(placeholder, cv.CV_LOAD_IMAGE_COLOR);
      cv.cvtColor(img, img, cv.COLOR_RGBA2RGB, 0);
      
      // Reset masks for all classes
      drawingsByClass.setOrig(img);


      let canvas = document.getElementById('canvas_img');
      // Set canvas dims to avoid error on first load
      canvas.width = img.cols;
      canvas.height = img.rows;
      canvas.style.width = img.cols;
      canvas.style.height = img.rows;
      cv.imshow('canvas_img', img);
      drawings.updateAll();
      drawings.updateFused();
      updateColorDot();
    });
}

function saveImage(){
  if (drawingsByClass.orig !== null){
    let ret = drawingsByClass.getFusedResult();
    // load cv::Mat into canvas in order to save it
    cv.imshow('invisible_canvas', ret);
    let href = document.getElementById("invisible_canvas").toDataURL();

    var el = document.createElement('a');
    el.setAttribute('href', href);
    el.setAttribute('download', drawingsByClass.name);
    el.style.display = 'none';
    document.body.appendChild(el);
    el.click();
    document.body.removeChild(el);
  } else {
    print("No image to save");
  }
}

// Functions to save and load mask from one image to another
var savedMask = null;
function copyMask(){
  if (drawingsByClass !== null){
    if (savedMask !== null){
      for(let cl in savedMask){
        if(savedMask[cl].annot !== undefined) savedMask[cl].annot.delete();
        if(savedMask[cl].origMask !== undefined) savedMask[cl].origMask.delete();
        if(savedMask[cl].mask !== undefined) savedMask[cl].mask.delete();
      }
    }
    savedMask = drawingsByClass.copyMask();
    print(savedMask);
  }
}
function pasteMask(){
  if (drawingsByClass !== null){
    drawingsByClass.pasteMask(savedMask);
    drawings.updateAll();
  }
}

function ord(str){return str.charCodeAt(0);}
function onKeyDown(e){
  let k = e.keyCode
  print(k)
  if ( k == ord('0') || k == 96) { // BG drawing
    print_info(" mark background regions with left mouse button \n")
    drawings.drawColors = drawingConstants.bG;
    k = 13; updateColorDot();
  } else if ( k == ord('1') || k == 97) { // FG drawing
    print_info(" mark foreground regions with left mouse button \n")
    drawings.drawColors = drawingConstants.fG;
    k = 13; updateColorDot();
  } else if ( k == ord('2') || k == 98) { // PR_BG drawing
    print_info(" mark PROBABLE background regions with left mouse button \n")
    drawings.drawColors = drawingConstants.bGProb;
    k = 13;updateColorDot();
  } else if ( k == ord('3') || k == 99) { // PR_FG drawing
    print_info(" mark PROBABLE foreground regions with left mouse button \n")
    drawings.drawColors = drawingConstants.fGProb;
    k = 13; updateColorDot();
  } else if ( k == ord('4') || k == 100) { // Eraser
    print_info(" mark PROBABLE foreground regions with left mouse button \n")
    drawings.drawColors = drawingConstants.eraser;
    k = 13; updateColorDot();
  } else if ( k == ord('S')) { // save image
    saveImage();
  } else if ( k == ord('R')) { // reset everything
    print_info("resetting \n");
    resetDrawings(drawings);
    drawings.updateAll();
  }
  if (k == ord('N') || k == 13) {
    if (drawings.rectOrMask == 0){ // Use rect only
      let bgdModel = new cv.Mat();
      let fgdModel = new cv.Mat();
      cv.grabCut(drawings.orig, drawings.mask, drawings.rect, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_RECT);
      bgdModel.delete(); fgdModel.delete();
    } else if (drawings.rectOrMask == 1){ // Use mask (for refinements)
      print("using the new parts");
      let bgdModel = new cv.Mat();
      let fgdModel = new cv.Mat();
      cv.grabCut(drawings.orig, drawings.mask, drawings.rect, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_MASK);
      bgdModel.delete(); fgdModel.delete();
    }
    drawings.updateAll();
    drawingsByClass.updateFused();
    print("Done");
  }
}

function getMousePos(e) {
  let canvas = drawings.imgCanvas;
  let rect = canvas.getBoundingClientRect();
  return {
      x: (e.clientX - rect.left) / scale,
      y: (e.clientY - rect.top) / scale
  };
}
function onMouseMove(e){
  if (drawings.rectangle){
    drawings.drawRectangle(getMousePos(e));
    drawings.rectOrMask = 0;
  } else if (drawings.drawing){
    drawings.drawLine(getMousePos(e));
    drawings.rectOrMask = 1;
  }
  
}

function onMouseDown(e){
  if (e.button==2){
    drawings.rectangle = true;
    drawings.rectStart = getMousePos(e);
  } else if (e.button==0){
    if (!drawings.rectDrawn){
      print_info("Draw a rectangle first");
    } else{
      drawings.drawing = true;
      drawings.prevLinePoint = getMousePos(e);
    }
  }
}

function onMouseUp(e){
  if (e.button==2){
    if (drawings.rectangle){
      drawings.rectangle = false;
      drawings.rectDrawn = true;
      drawings.drawRectangle(getMousePos(e));
      drawings.rectOrMask = 0;
      print_info(" Now press the key 'n' a few times until no further change");
      
      onKeyDown({'keyCode': ord('N')}); // automatic segmentation
      // Save original mask
      if (drawings.origMask == null)  drawings.origMask = drawings.mask.clone();
      else                            drawings.mask.copyTo(drawings.origMask);
      
    }
  } else if (e.button==0){
    if (!drawings.rectDrawn){
      print_info("Draw a rectangle first");
    } else if (drawings.drawing) {
      drawings.drawing = false;
      drawings.drawLine(getMousePos(e));
      
      // Change 'hidden' mask in case the eraser was used
      if (drawings.drawColors == drawingConstants.eraser){
        let newMmask = drawings.origMask.clone();
        // Flatten annot to single channel
        let annotMask = new cv.Mat.zeros(drawings.annot.rows, drawings.annot.cols, cv.CV_8UC1);
        cv.cvtColor(drawings.annot, annotMask, cv.COLOR_RGB2GRAY);
        // Copy over ONLY the MANUALLY annotated values
        drawings.mask.copyTo(newMmask, annotMask);
        drawings.mask.delete();
        annotMask.delete();

        drawings.mask = newMmask;
      }

      // onKeyDown({'keyCode': ord('N')}); // automatic segmentation
    }
  }
}

let scrollOrder = ["bG", "fG", "bGProb", "fGProb", "eraser"];
function onMouseWheel(e){
  e.preventDefault();
  const direction = -Math.sign(e.deltaY);
  const oldInd = drawings.drawColors.index;

  const newInd = Math.min(4, Math.max(0, oldInd+direction));
  drawings.drawColors = drawingConstants[scrollOrder[newInd]];
  updateColorDot();
  onKeyDown({'keyCode': ord('N')}); // automatic segmentation
}

const colorDot = document.getElementById("color_dot");
const colorDotLabel = document.getElementById("color_dot_label");
const colorNames = ["Certain Background", "Certain Foreground", "Probable Background",
                    "Probable Foreground", "Eraser"];
function updateColorDot(){
  if (drawings !== undefined){ // edge case at loading time
    colorDot.style.backgroundColor = rgb2Hex(drawings.drawColors.color);
    colorDotLabel.innerHTML = colorNames[drawings.drawColors.index];
  }
}


// Seperate showing status to be global thus NOT attached to a specific drawing
const imgShow = document.getElementsByName("img_show");
let imgShowStatus = {}
for (let i in imgShow){
  let checkBox = imgShow[i];
  imgShowStatus[checkBox.value] = checkBox.checked;
}
 
function changeImgShow() {
  for (let i in imgShow){
    let checkBox = imgShow[i];
    imgShowStatus[checkBox.value] = checkBox.checked;
  }
  drawings.updateImg();
  drawingsByClass.updateFused();
}