import React from 'react';
import * as d3 from 'd3';
import { requestAPI } from './gquantlab';
// import YAML from 'yaml';
// import Moment from 'moment';

import {
  handleMouseOver,
  handleMouseOut,
  handleMouseLeft,
  handleClicked,
  handleMouseMoved,
  handleHighlight,
  handleDeHighlight
} from './eventHandler';
import { drag } from './dragHandler';
import { handleMouseDown, handleMouseUp } from './connectionHandler';
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import NodeEditor from './nodeEditor';
import { validConnection } from './validator';
import { INode, IEdge, ContentHandler } from './document';
import { exportWorkFlowNodes } from './chartEngine';
import { OUTPUT_COLLECTOR } from './mainComponent';

interface IPortInfo {
  [key: string]: any;
}

interface IPoint {
  x: number;
  y: number;
  id: string;
}
interface IMappedEdge {
  source: IPoint;
  target: IPoint;
}

interface IChartState {
  addMenu: boolean;
  x: number;
  y: number;
  opacity: number;
  nodeDatum: any;
}

interface IChartProp {
  contentHandler: ContentHandler;
  nodes: INode[];
  edges: IEdge[];
  setChartState: Function;
  width: number;
  height: number;
  layout: Function;
}

export class Chart extends React.Component<IChartProp, IChartState> {
  myRef: React.RefObject<HTMLDivElement>;
  mouse: { x: number; y: number };
  mousePage: { x: number; y: number };
  starting: { groupName: string; from: string; point: IPoint };
  tooltip: any;
  textHeight: number;
  circleHeight: number;
  circleRadius: number;
  bars: any;
  svg: any;
  transform: any;
  link: any;
  mouseLink: any;
  g: any;
  inputPorts: Set<string>;
  outputPorts: Set<string>;
  inputRequriements: { [key: string]: any };
  outputColumns: { [key: string]: any };
  portTypes: { [key: string]: string[] };
  isDirty: boolean;

  constructor(props: IChartProp) {
    super(props);
    this.myRef = React.createRef<HTMLDivElement>();
    this.mouse = null;
    this.mousePage = null;
    this.starting = null;
    this.tooltip = null;
    this.textHeight = 25;
    this.circleHeight = 20;
    this.circleRadius = 8;
    this.bars = null;
    this.svg = null;
    this.transform = null;
    this.link = null;
    this.mouseLink = null;
    this.g = null;
    this.isDirty = false;
    this.state = {
      addMenu: true,
      x: -1000,
      y: -1000,
      opacity: 0,
      nodeDatum: null
    };
    this.inputPorts = new Set();
    this.outputPorts = new Set();
    this.inputRequriements = {};
    this.outputColumns = {};
    this.portTypes = {};
    this.props.contentHandler.reLayoutSignal.connect(this.nodeReLayout, this);
    this.props.contentHandler.nodeAddedSignal.connect(this.addNewNode, this);
  }

  portMap(): void {
    this.inputPorts = new Set();
    this.outputPorts = new Set();
    this.inputRequriements = {};
    this.outputColumns = {};
    this.props.nodes.forEach((d: INode) => {
      const nodeId = d.id;
      d.inputs.forEach(k => {
        this.inputPorts.add(nodeId + '.' + k.name);
        this.portTypes[nodeId + '.' + k.name] = k.type;
      });
      d.outputs.forEach(k => {
        this.outputPorts.add(nodeId + '.' + k.name);
        this.portTypes[nodeId + '.' + k.name] = k.type;
      });
      let keys = Object.keys(d.required);
      keys.forEach(k => {
        this.inputRequriements[nodeId + '.' + k] = d.required[k];
      });
      keys = Object.keys(d.output_columns);
      keys.forEach(k => {
        this.outputColumns[nodeId + '.' + k] = d.output_columns[k];
      });
    });
  }

  translateCorr(portStr: string, from = false): IPoint {
    const splits = portStr.split('.');
    const nodeId = splits[0];
    const outputPort = splits[1];
    const nodeObj = this.props.nodes.filter(d => d.id === nodeId)[0];
    if (from) {
      const index = nodeObj.outputs.findIndex(d => d.name === outputPort);
      const x = nodeObj.x + nodeObj.width;
      const y = nodeObj.y + (index + 0.4) * this.circleHeight + this.textHeight;
      const point = { x: x, y: y, id: nodeId };
      return point;
    } else {
      const index = nodeObj.inputs.findIndex(d => d.name === outputPort);
      const x = nodeObj.x;
      const y = nodeObj.y + (index + 0.4) * this.circleHeight + this.textHeight;
      const point = { x: x, y: y, id: nodeId };
      return point;
    }
  }

  edgeMap(d: IEdge): IMappedEdge {
    const sourcePoint = this.translateCorr(d.from, true);
    const targetPoint = this.translateCorr(d.to, false);
    return { source: sourcePoint, target: targetPoint };
  }

  edgeData(): IMappedEdge[] {
    return this.props.edges.map(this.edgeMap.bind(this));
  }

  nodeReLayout(sender: ContentHandler, inputs: void): void {
    this.reLayout();
  }

  addNewNode(sender: ContentHandler, inputs: INode): void {
    const result: INode = JSON.parse(JSON.stringify(inputs));
    // if the Output Collector is already added, ignore it
    const findDup = this.props.nodes.findIndex((d:INode)=> d.id === OUTPUT_COLLECTOR);
    console.log(findDup);
    if ( findDup >= 0){
      return;
    }
    if (result.type === 'Output Collector') {
      result.id = OUTPUT_COLLECTOR;
    }
    else{
      result.id = Math.random()
        .toString(36)
        .substring(2, 15);
    }
    result.x = this.mouse ? this.mouse.x : 0;
    result.y = this.mouse ? this.mouse.y : 0;
    this.props.nodes.push(result);
    this.props.setChartState({
      nodes: this.props.nodes,
      edges: this.props.edges
    });
  }

  componentWillUnmount(): void {
    this.props.contentHandler.reLayoutSignal.disconnect(
      this.nodeReLayout,
      this
    );
    this.props.contentHandler.nodeAddedSignal.disconnect(this.addNewNode, this);
  }

  componentDidMount(): void {
    this.tooltip = d3
      .select(this.myRef.current)
      .append('div')
      .attr('class', 'tooltip')
      .style('opacity', 0);

    const zoom = d3
      .zoom()
      .scaleExtent([0.1, 30])
      .on('zoom', d => {
        this.g.attr('transform', d3.event.transform);
        this.transform = d3.event.transform;
      });

    this.svg = d3
      .select(this.myRef.current)
      .append('svg')
      .attr('width', this.props.width ? this.props.width : 100)
      .attr('height', this.props.height ? this.props.height : 100)
      .attr(
        'viewBox',
        `0 0 ${(this.props.width ? this.props.width : 0) + 300} ${(this.props
          .height
          ? this.props.height
          : 0) + 300}`
      )
      .attr('font-family', 'sans-serif')
      .attr('font-size', '14')
      .attr('text-anchor', 'end')
      .on('mouseleave', handleMouseLeft(this))
      .on('mousemove', handleMouseMoved(this))
      .on('click', handleClicked(this))
      //      .on('contextmenu', handleRightClick(this))
      .call(zoom.bind(this));

    //d3.select("body").on("keydown", handleKey(this));
    this.g = this.svg.append('g');

    this.bars = this.g.selectAll('g').data(this.props.nodes);

    this.link = this.g
      .append('g')
      .attr('stroke', '#999')
      .selectAll('line')
      .data(this.edgeData())
      .join('line');

    this.mouseLink = this.g
      .append('g')
      .attr('stroke', 'red')
      .selectAll('line');
  }

  drawCircles(): void {
    const portsInput = this.bars
      .selectAll('g')
      .filter((d: INode, i: number) => i === 0)
      .data((d: INode, i: number) => [d])
      .join('g')
      .attr(
        'transform',
        (d: INode, i: number) => `translate(0, ${this.textHeight})`
      )
      .attr('group', 'inputs');

    portsInput
      .selectAll('circle')
      .data((d: INode) => {
        const data = [];
        for (let i = 0; i < d.inputs.length; i++) {
          if (d.inputs[i].name in d.required) {
            const portInfo: IPortInfo = {};
            portInfo['content'] = d.required[d.inputs[i].name];
            portInfo['portType'] = d.inputs[i].type;
            data.push({
              [d.id + '.' + d.inputs[i].name]: portInfo
            });
          } else {
            const portInfo: IPortInfo = {};
            portInfo['content'] = {};
            portInfo['portType'] = d.inputs[i].type;
            data.push({
              [d.id + '.' + d.inputs[i].name]: portInfo
            });
          }
        }
        return data;
      })
      .join('circle')
      .attr('fill', (d: any) => {
        if (!this.starting) {
          return 'blue';
        }
        const key = Object.keys(d)[0];
        if (validConnection(this)(this.starting.from, key)) {
          return 'blue';
        } else {
          return 'white';
        }
      })
      .attr('cx', 0)
      .attr('cy', (d: any, i: number) => (i + 0.4) * this.circleHeight)
      .attr('r', this.circleRadius)
      .on('mouseover', handleMouseOver(this))
      .on('mouseout', handleMouseOut(this))
      .on('mousedown', handleMouseDown(this))
      .on('mouseup', handleMouseUp(this));

    const portsOutput = this.bars
      .selectAll('g')
      .filter((d: INode, i: number) => i === 1)
      .data((d: INode, i: number) => [d])
      .join('g')
      .attr(
        'transform',
        (d: INode, i: number) =>
          `translate(${d ? d.width : 0}, ${this.textHeight})`
      )
      .attr('group', 'outputs');

    portsOutput
      .selectAll('circle')
      .data((d: INode) => {
        const data = [];
        for (let i = 0; i < d.outputs.length; i++) {
          if (d.outputs[i].name in d.output_columns) {
            const portInfo: IPortInfo = {};
            portInfo['content'] = d.output_columns[d.outputs[i].name];
            portInfo['portType'] = d.outputs[i].type;
            data.push({
              [d.id + '.' + d.outputs[i].name]: portInfo
            });
          } else {
            const portInfo: IPortInfo = {};
            portInfo['content'] = {};
            portInfo['portType'] = d.outputs[i].type;
            data.push({
              [d.id + '.' + d.outputs[i].name]: portInfo
            });
          }
        }
        return data;
      })
      .join('circle')
      .attr('fill', (d: any) => {
        if (!this.starting) {
          return 'green';
        }
        const key = Object.keys(d)[0];
        if (validConnection(this)(this.starting.from, key)) {
          return 'green';
        } else {
          return 'white';
        }
      })
      .attr('cx', 0)
      .attr('cy', (d: any, i: number) => (i + 0.4) * this.circleHeight)
      .attr('r', this.circleRadius)
      .on('mouseover', handleMouseOver(this))
      .on('mouseout', handleMouseOut(this))
      .on('mousedown', handleMouseDown(this))
      .on('mouseup', handleMouseUp(this));
  }

  drawLinks(): void {
    this.link = this.link
      .data(this.edgeData())
      .join('line')
      .attr('x1', (d: IMappedEdge) => d.source.x)
      .attr('y1', (d: IMappedEdge) => d.source.y)
      .attr('x2', (d: IMappedEdge) => d.target.x)
      .attr('y2', (d: IMappedEdge) => d.target.y);
  }

  componentDidUpdate(): void {
    this.bars = this.bars
      .data(this.props.nodes)
      .join('g')
      .attr(
        'transform',
        (d: INode, i: number) => `translate(${d ? d.x : 0}, ${d ? d.y : 0})`
      );

    this.bars
      .selectAll('rect')
      .filter((d: INode, i: number) => i === 0)
      .data((d: INode, i: number) => [
        {
          w: d.width,
          h:
            this.textHeight +
            Math.max(d.inputs.length, d.outputs.length) * this.circleHeight
        }
      ])
      .join('rect')
      .attr('fill', 'steelblue')
      .attr('width', (d: { w: number; h: number }) => d.w)
      .attr('height', (d: { w: number; h: number }) => d.h)
      .on('mouseover', handleHighlight(this, 'red', 'pointer'))
      .on('mouseout', handleDeHighlight(this));

    this.bars
      .selectAll('rect')
      .filter((d: INode, i: number) => i === 1)
      .data((d: INode, i: number) => [{ w: d.width, h: this.textHeight }])
      .join('rect')
      .attr('fill', 'seagreen')
      .attr('width', (d: { w: number; h: number }) => d.w)
      .attr('height', (d: { w: number; h: number }) => d.h)
      .call(drag(this.props.setChartState, this))
      .on('mouseover', handleHighlight(this, 'black', 'grab'))
      .on('mouseout', handleDeHighlight(this));

    this.bars
      .selectAll('text')
      .filter((d: INode, i: number) => i === 0)
      .data((d: INode, i: number) => [{ w: d.width, id: (d.id===OUTPUT_COLLECTOR?"":d.id)}])
      .join('text')
      .attr('fill', 'white')
      .attr('x', (d: { w: number; id: string }) => d.w)
      .attr('y', 0)
      .attr('dy', '1.00em')
      .attr('dx', '-1.00em')
      .text((d: { w: number; id: string }) => d.id)
      .call(drag(this.props.setChartState, this))
      .on('mouseover', handleHighlight(this, 'black', 'grab'))
      .on('mouseout', handleDeHighlight(this));

    this.bars
      .selectAll('text')
      .filter((d: INode, i: number) => i === 1)
      .data((d: INode, i: number) => [{ w: d.width, text: d.type }])
      .join('text')
      .attr(
        'transform',
        (d: { w: number; text: string }, i: number) =>
          `translate(0, ${this.textHeight})`
      )
      .attr('fill', 'black')
      .attr('x', (d: { w: number; text: string }, i: number) => d.w)
      .attr('y', 0)
      .attr('dy', '1.00em')
      .attr('dx', '-1.00em')
      .text((d: { w: number; text: string }, i: number) => d.text)
      .on('mouseover', handleHighlight(this, 'red', 'pointer'))
      .on('mouseout', handleDeHighlight(this));

    this.drawCircles();
    this.drawLinks();
  }

  reLayout(): void {
    this.props.layout(this.props.nodes, this.props.edges, this.transform);
  }

  updateInputs(json: string): void {
    /**
     * send the taskgraph to backend to run the column-flow logics so all the output types and names are computed
     */
    const workflows = requestAPI<any>('load_graph', {
      body: json,
      method: 'POST'
    });

    workflows.then((data: any) => {
      const newNode: {
        [key: string]: { required: any; outputColumns: any };
      } = {};
      data.nodes.forEach((d: INode) => {
        newNode[d.id] = {
          required: d.required,
          outputColumns: d.output_columns
        };
      });
      this.props.nodes.forEach((d: INode) => {
        if (d.id in newNode) {
          d.required = newNode[d.id].required;
          // eslint-disable-next-line @typescript-eslint/camelcase
          d.output_columns = newNode[d.id].outputColumns;
        }
      });
      this.props.setChartState({
        nodes: this.props.nodes,
        edges: this.props.edges
      });
    });
  }

  configFile(): INode[] {
    return exportWorkFlowNodes(this.props.nodes, this.props.edges);
  }

  render(): JSX.Element {
    this.portMap();
    if (this.svg) {
      this.svg
        .attr('width', this.props.width ? this.props.width : 100)
        .attr('height', this.props.height ? this.props.height : 100)
        .attr(
          'viewBox',
          `0 0 ${(this.props.width ? this.props.width : 0) + 300} ${(this.props
            .height
            ? this.props.height
            : 0) + 300}`
        );
    }

    console.log('rendering');
    if (this.state.addMenu) {
      return <div ref={this.myRef} />;
    } else {
      return (
        <div ref={this.myRef}>
          <NodeEditor
            x={this.state.x}
            y={this.state.y}
            opacity={this.state.opacity}
            nodeDatum={this.state.nodeDatum}
            setChartState={this.props.setChartState}
            nodes={this.props.nodes}
            edges={this.props.edges}
            setMenuState={this.setState.bind(this)}
          />
        </div>
      );
    }
  }
}
