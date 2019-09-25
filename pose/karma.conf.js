process.env.CHROME_BIN = require('puppeteer').executablePath();

module.exports = function(config) {
  config.set({
    frameworks: ['mocha', 'karma-typescript'],
    files: [
      {pattern: 'src/**/*.ts'},
      {pattern: 'test/**/*.ts'},
      {pattern: 'test/**/*.html'},
      {pattern: 'dist/**/*.min.js'}
    ],
    preprocessors: {
      '**/*.ts': ['karma-typescript'],  // *.tsx for React Jsx
    },
    karmaTypescriptConfig: {
      tsconfig: 'tsconfig.json',
      // we include this here to keep our test files separate,
      // but not include tests in normal builds
      include: [ 'test/**/*.ts'],
      reports: {} // Do not produce coverage html.
    },
    //logLevel: config.LOG_DEBUG,
    reporters: ['progress', 'karma-typescript'],
    browsers: ['ChromeHeadless'], // ['Chrome'/*, 'Firefox'*/],
    reportSlowerThan: 500,
    browserNoActivityTimeout: 30000
  });
};
