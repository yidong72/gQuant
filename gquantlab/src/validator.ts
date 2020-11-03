import { IEdge } from './document';
import { Chart } from './chart';

//TODO, need to refactor this to handle dynamcially added module type checks 
function valid(required: any, outputs: any): boolean {
  const keys = Object.keys(required);
  for (let i = 0; i < keys.length; i++) {
    const d = keys[i];
    if (
      !(
        d in outputs &&
        (outputs[d] === null ||
          required[d] === null ||
          required[d] === outputs[d])
      )
    ) {
      return false;
    }
  }
  return true;
}

function validNmType(required: any, outputs: any): boolean {
  //first check types

  const reqElement = required['element'];
  const outElement = outputs['element'];
  if (
    outElement['types'][0] !== 'VoidType' &&
    reqElement['types'][0] !== 'VoidType'
  ) {
    if (
      outElement['types'].findIndex(
        (d: string) => d === reqElement['types'][0]
      ) < 0
    ) {
      // req type should be generic,
      // out type should be specific, i.e. subclass of req
      // first required element type should be the parent type of the output element
      return false;
    }
    if (outElement['fields'] !== reqElement['fields']) {
      return false;
    }
    if (outElement['parameters'] !== reqElement['parameters']) {
      return false;
    }
  }

  const reqAxes = required['axes'];
  const outAxes = outputs['axes'];
  if (reqAxes.length === 0) {
    return true;
  }
  if (reqAxes.length !== outAxes.length) {
    return false;
  }
  for (let i = 0; i < reqAxes.length; i++) {
    if (reqAxes[i]['kind'] !== outAxes[i]['kind']) {
      return false;
    }
    if (
      reqAxes[i]['size'] !== null &&
      outAxes[i]['size'] !== null &&
      reqAxes[i]['size'] !== outAxes['size']
    ) {
      return false;
    }
  }
  return true;
}

export function validConnection(that: Chart) {
  return function(from: string, to: string): any {
    if (from === to) {
      return true;
    }
    //canot connect to the port from the same node
    if (from.split('.')[0] === to.split('.')[0]) {
      return false;
    }

    // check whether from is input or output
    const fromOutputPorts = that.outputPorts.has(from);
    if (fromOutputPorts) {
      // to has to be an input port
      if (!that.inputPorts.has(to)) {
        return false;
      }
      //from a output port, multiple connection from the same output port is allowed,
      if (that.props.edges.findIndex((d: IEdge) => d.to === to) >= 0) {
        // if there is already a connection to the input port, it is not valid
        return false;
      }

      const toTypes = that.portTypes[to];
      const fromTypes = that.portTypes[from];
      // if 'any shows up in types, it is valid
      if (
        toTypes.findIndex(d => d === 'any') < 0 &&
        fromTypes.findIndex(d => d === 'any') < 0
      ) {
        const intersection = toTypes.filter((x: string) =>
          fromTypes.includes(x)
        );
        if (intersection.length === 0) {
          return false;
        }
      }

      // make sure the requirement is met
      if (from in that.outputColumns && to in that.inputRequriements) {
        const nmType = toTypes.findIndex(d => {
          return d.indexOf('NmTensor') >= 0;
        });
        if (nmType >= 0) {
          return validNmType(
            that.inputRequriements[to],
            that.outputColumns[from]
          );
        } else {
          return valid(that.inputRequriements[to], that.outputColumns[from]);
        }
      } else if (
        !(from in that.outputColumns) &&
        to in that.inputRequriements
      ) {
        return valid(that.inputRequriements[to], {});
      }
    } else {
      //from a input port, only single connection from input port is allowed
      // to has to be output port
      if (!that.outputPorts.has(to)) {
        return false;
      }
      if (that.props.edges.findIndex((d: IEdge) => d.to === from) >= 0) {
        // if there is already a connection to the input port, it is not valid
        return false;
      }
      const toTypes = that.portTypes[to];
      const fromTypes = that.portTypes[from];
      if (
        toTypes.findIndex(d => d === 'any') < 0 &&
        fromTypes.findIndex(d => d === 'any') < 0
      ) {
        const intersection = toTypes.filter((x: string) =>
          fromTypes.includes(x)
        );
        if (intersection.length === 0) {
          return false;
        }
      }

      // make sure the requirement is met
      if (to in that.outputColumns && from in that.inputRequriements) {
        const nmType = toTypes.findIndex(d => {
          return d.indexOf('NmTensor') >= 0;
        });
        if (nmType >= 0) {
          return validNmType(
            that.inputRequriements[from],
            that.outputColumns[to]
          );
        } else {
          return valid(that.inputRequriements[from], that.outputColumns[to]);
        }
      } else if (
        !(to in that.outputColumns) &&
        from in that.inputRequriements
      ) {
        return valid(that.inputRequriements[from], {});
      }
    }
    return true;
  };
}
