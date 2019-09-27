// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const { resolve } = require('path')
const cloneDeep = require('lodash.clonedeep');
const TerserPlugin = require('terser-webpack-plugin');
const outputPath = resolve('dist');

/**
 * This is the base Webpack Config
 * depending on options, such as --mode=production, the config will be altered
 * each time it executes.
 */
const baseConfig = {
    entry : './src/index.ts',
    output : {
        path : outputPath,
        library: ['tmPose'],
        filename : 'teachablemachine-pose.min.js'
    },
    mode : 'development',
    watchOptions: {
        ignored: /src\/version\.ts/
    },
    module : {
        rules : [
            {
                test: /\.js$/,
                use: ["source-map-loader"],
                enforce: "pre"
            },
            {
                test : /\.tsx?$/,
                use : 'ts-loader',
                exclude : /node_modules/
            }
        ]
    },
    resolve : {
        extensions : ['.tsx', '.ts', '.js']
    },
    externals: {
        "@tensorflow/tfjs": "tf"
    },
    devtool : 'inline-source-map'
}

module.exports = function(opts, argv) {

    const { bold, green } = colorizer(argv.color && typeof argv.color === 'function' && argv.color().hasBasic === true);

    console.log(`${bold('Mode')}: ${green(argv.mode)}`);
    console.log(`${bold('Reporter')}:  ${green(argv.reporter)}`);

    const config = cloneDeep(baseConfig);

    if (argv.mode === 'production'){
        config.output.path = resolve(`${outputPath}`);
        //turn off source maps
        config.devtool = 'none';

        config.optimization = {
            minimizer: [
                new TerserPlugin({
                    parallel: true,
                    cache: './.terser_build_cache',
                    //exclude: /transpiledLibs/,
                    terserOptions: {
                      warnings: false,
                      ie8: false
                    }
                  })
            ]
        };
    }

    return [config];

}


/**
 * Simple Utility for adding color to logs
 * @param {boolean} supported
 */
function colorizer(supported) {
    if (supported) {
        return {
            green: (str) => `\u001b[32m${str}\u001b[39m`,
            bold: (str) => `\u001b[1m${str}\u001b[22m`
        };
    }

    const identity = (str) => str;

    return {
        green: identity,
        bold: identity
    };
};
