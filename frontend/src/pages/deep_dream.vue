<template>
    <div class="page">
        <div class="menu">
            <div class="item">
                <div class="title">model</div>
                <el-select class='value' size="small" v-model="params.model" @change="getLayers">
                    <el-option v-for="model in models" :key="model" :value='model'/>
                </el-select>
            </div>
            <div class="item">
                <div class="title">input</div>
                <el-select class="value" size="small" v-model="params.input" @change="update">
                    <el-option value='static/images/cat_dog.png'/>
                    <el-option value='static/images/spider.png'/>
                    <el-option value='static/images/dd_tree.jpg'/>
                </el-select>
            </div>
            <div class="item">
                <div class="title">layer</div>
                <el-select class="value" size="small" v-model="params.layer" @change="update">
                    <el-option v-for="(layer, index) in layers" :value="index" :key="index">{{index}} - {{layer.name}}/{{layer.layer}}</el-option>
                </el-select>
            </div>            <div class="item"><div class="title">activation</div><el-input class='value' size="small" v-model="params.activation"  @change="update"/></div>
            <div class="item"><div class="title">epochs</div><el-input class='value' size="small" v-model="params.epochs"  @change="update"/></div>
            <div class="item"><div class="title">lr</div><el-input class='value' size="small" v-model="params.lr"  @change="update"/></div>
            <div class="item"><el-checkbox class='button' v-model="params.clamp" @change="update">clamp</el-checkbox></div>
        </div>
        <div class="network">
            <div class="iter">
                <img :src="res.output" width="256" height="256"/>
                <div>epoch = {{res.epoch}}, loss = {{res.loss}}</div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    data() {
        return {
            models: [],
            layers: {},
            res: {},
            params: {
                model: 'vgg19',
                input: 'static/images/dd_tree.jpg',
                layer: 34,
                activation: 94,
                epochs: 251,
                lr: 12,
                clamp: false,
            }
        };
    },
    created() {
        this.config()
        this.update()
    },
    sockets: {
        models(data) {
            this.models = data
        },
        layers(data) {
            this.layers = data
        },
        response_deep_dream(data) {
            this.res = data
        }
    },
    methods: {
        config() {
            this.$socket.emit('get_models')
            this.getLayers()
        },
        getLayers() {
            this.$socket.emit('get_layers', { model: this.params.model })
        },
        update() {
            this.$socket.emit("deep_dream", this.params);
        },
    }
};
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
.page {
    .network {
        .iter {
            display: flex;
            flex-flow: column;
            align-items: center;
        }
    }
}

</style>
